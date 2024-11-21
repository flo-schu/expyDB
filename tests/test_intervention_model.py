import pytest
import numpy as np

from expyDB.intervention_model import (
    Experiment, 
    Treatment, 
    Timeseries, 
    TsData,
    from_expydb,
)
from expyDB.database_operations import (
    create_database, 
    experiment_to_db,
)


# @pytest.fixture(scope="session")
def test_model_to_database(tmp_path):
    database = tmp_path / "test.db"
    create_database(database=database, force=True)

    experiment = Experiment.model_validate({})
    treatment = Treatment.model_validate({})
    timeseries_intervention = Timeseries.model_validate(dict(type="intervention", variable="oxygen", unit="mg/L"))
    timeseries_observation = Timeseries.model_validate(dict(type="observation", variable="respiration", unit="mg/L"))
    treatment.interventions.append(timeseries_intervention)
    treatment.observations.append(timeseries_observation)
    experiment.treatments.append(treatment)
    
    time_intervention = np.arange(0, 11, step=5, dtype="timedelta64[h]")
    oxygen_ts = np.array([0, 5.0, 5.0])
    
    time_observation = np.linspace(0, 10, 51, dtype="timedelta64[h]")
    respiration_ts = np.linspace(5, 3, 51)

    tsdata_interventions = [
        TsData.model_validate(TsData(time=ti, value=vi)) 
        for ti, vi in zip(time_intervention, oxygen_ts)
    ]

    tsdata_observations = [
        TsData.model_validate(TsData(time=ti, value=vi)) 
        for ti, vi in zip(time_observation, respiration_ts)
    ]
    timeseries_intervention.tsdata.extend(tsdata_interventions)
    timeseries_observation.tsdata.extend(tsdata_observations)

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


if __name__ == "__main__":
    from pathlib import Path
    test_model_to_database(Path("/tmp"))
    test_from_db(Path("/tmp"))


