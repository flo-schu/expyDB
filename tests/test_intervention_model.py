import pytest
import numpy as np
import pandas as pd

from expyDB.intervention_model import (
    Experiment, 
    Treatment, 
    Timeseries, 
    TsData,
    from_expydb,
    PandasConverter
)
from expyDB.database_operations import (
    create_database, 
    experiment_to_db,
)
from sqlmodel import (
    Session,
    create_engine,
    select
)



# @pytest.fixture(scope="session")
def test_model_to_database(tmp_path):
    database = tmp_path / "test.db"
    create_database(database=database, force=True)

    ts_meta_int = dict(type="intervention", variable="oxygen", unit="mg/L", time_unit="hour")
    ts_meta_obs = dict(type="observation", variable="respiration", unit="mg/L", time_unit="hour")

    experiment = Experiment.model_validate({})
    treatment = Treatment.model_validate({})
    timeseries_intervention_1 = Timeseries.model_validate(ts_meta_int)
    timeseries_intervention_2 = Timeseries.model_validate(ts_meta_int)
    timeseries_observation_1 = Timeseries.model_validate(ts_meta_obs)
    timeseries_observation_2 = Timeseries.model_validate(ts_meta_obs)
    treatment.timeseries.append(timeseries_intervention_1)
    treatment.timeseries.append(timeseries_intervention_2)
    treatment.timeseries.append(timeseries_observation_1)
    treatment.timeseries.append(timeseries_observation_2)
    experiment.treatments.append(treatment)
    
    time_intervention = np.arange(0, 11, step=5, dtype="timedelta64[h]")
    oxygen_ts = np.array([0, 5.0, 5.0])
    
    time_observation = np.linspace(0, 10, 51, dtype="timedelta64[h]")
    respiration_ts_1 = np.linspace(5, 3, 51)
    respiration_ts_2 = np.linspace(5, 3, 51) + 1


    tsdata_interventions_1 = [
        TsData.model_validate(TsData(time=ti, value=vi)) 
        for ti, vi in zip(time_intervention, oxygen_ts)
    ]
    tsdata_interventions_2 = [
        TsData.model_validate(TsData(time=ti, value=vi)) 
        for ti, vi in zip(time_intervention, oxygen_ts)
    ]
    timeseries_intervention_1.tsdata = tsdata_interventions_1
    timeseries_intervention_2.tsdata = tsdata_interventions_2

    tsdata_observations_1 = [
        TsData.model_validate(TsData(time=ti, value=vi)) 
        for ti, vi in zip(time_observation, respiration_ts_1)
    ]

    tsdata_observations_2 = [
        TsData.model_validate(TsData(time=ti, value=vi)) 
        for ti, vi in zip(time_observation, respiration_ts_2)
    ]
    timeseries_observation_1.tsdata = tsdata_observations_1
    timeseries_observation_2.tsdata = tsdata_observations_2

    # test init form excel file
    pandas_converter = PandasConverter(experiment)
    pandas_converter.to_excel(tmp_path / "test.xlsx")

    experiment_to_db(database=database, experiment=experiment)

def test_from_db(tmp_path):
    database = f"sqlite:///{tmp_path / 'test.db'}"
    observations, interventions = from_expydb(database)

    np.testing.assert_array_equal(
        observations.respiration.respiration, # type: ignore
        np.expand_dims(np.linspace(5, 3, 51), axis=0)
    )

    np.testing.assert_array_equal(
        observations.respiration.time, # type: ignore
        np.linspace(0, 10, 51, dtype="timedelta64[h]")    
    )

def test_get_timeseries(tmp_path):
    database = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(database, echo=True)
    with Session(engine) as session:
        statement = select(Timeseries).where(Timeseries.type == "intervention")
        results = session.exec(statement)
        results_list = results.all()
        ts = results_list[0]
        ts.model_dump()

            
        statement = select(Treatment)\
            .join(Timeseries)\
            .where(Timeseries.type=="intervention")\
            .join(Experiment)\
            .distinct() # distinct makes sure to get only unique treatments
        results = session.exec(statement)
        results_list = results.all()

        ts: Timeseries = results_list[0].interventions[0]

    

if __name__ == "__main__":
    from pathlib import Path
    test_model_to_database(Path("/tmp"))
    test_from_db(Path("/tmp"))
    test_get_timeseries(Path("/tmp"))


