import warnings
import os
from typing import List
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from database_model import Observation, Treatment, Experiment, Base
from timepath.helpers.misc import label_duplicates, replace_na

# add data
data = pd.read_csv("data/dataset_3/cint.csv")
data = data.dropna(how="all").copy()    
data = data.rename(columns={
    "Exp.ID": "experiment_id",
    "Exp.date": "experiment_date",
    "Cext.molL": "cext",
    "Cint.molL": "cint",
    "Cext.nom.molL": "cext_nom",
    "Cint.nM.ZFE": "cint_zfe",
    "Mix": "mix",
    "n_ZFE": "n_zfe"
}).drop(columns=[
    "Cext0.molL"
])
data["experiment_date"] = pd.to_datetime(data["experiment_date"], format="%m-%d-%y")
data["time"] = data.hpf + data.hpe
data["mix"] = np.where(data.mix == "y", True, False)

measurement_variables = {
    "cext": "mol L-1",
    "cint": "mol L-1",
}

# transpose dataframe to long format
data = data.melt(
    id_vars=[v for v in data.columns if v not in measurement_variables],
    value_vars=measurement_variables.keys(),
    value_name="value",
    var_name="measurement"
)

data["unit"] = data.measurement.map(measurement_variables)

# Fill missing values. This is necessary to avoid grouping chaos
data = replace_na(data, "experiment_date", default_value=pd.to_datetime("1900-01-01"))
data = replace_na(data, "experimentator", default_value="UNKNOWN")

experiment_variables = ["experiment_date", "experimentator"]
treatment_vars = ["substance", "mix", "cext_nom", "cext_nom_total", "hpf", "n_zfe"]

def treatment_setter(variables, identifiers):
    (substance, mix, cext_nom, cext_nom_tot, hpf, n_zfe) = identifiers
                    
    if not mix:
        exposure_map = {f"cext_nom_{substance.lower()}": cext_nom}
    else:
        # this is in information about the mixture that I got from 
        # 2018 and 2019 experiments
        exposure_map = {
            "cext_nom_diuron": cext_nom_tot * 0.11,
            "cext_nom_diclofenac": cext_nom_tot * 1000 * 0.026,
            "cext_nom_naproxen": cext_nom_tot * 1000 * 0.864,
        }

    return dict(
        hpf=hpf,
        nzfe=n_zfe,
        **exposure_map,
    )

def experiment_setter(variables, identifiers):
    experiment_date, experimentator = identifiers
    return {"date": experiment_date, "experimentator": experimentator}

def setter(variables, identifiers):
    return {key:value for key, value in zip(variables, identifiers)}

def add_data(
    database: str,
    data: pd.DataFrame,
    experiment_variables: List[str],
    treatment_variables: List[str],
    experiment_setter: callable = setter,
    treatment_setter: callable = setter,
):

    nans = data[treatment_vars].isna().values.sum(axis=0)
    if np.any(nans > 0):
        warnings.warn(
            f"NaNs in treatment variables {np.array(treatment_vars)[nans > 0]} detected. " 
            "Fix in data input, define default, or live with nans in treatment info."
        )

    # Create an engine to connect to your database
    CREATED_AT = datetime.datetime.now()
    engine = create_engine(f"sqlite:///{database}", echo=False)

    with Session(engine) as session:

        # group by experiment
        exp_groups = data.groupby(experiment_variables, dropna=False)
        for experiment_identifiers, experiment_rows in exp_groups:
            experiment = Experiment(
                created_at=CREATED_AT,
                **experiment_setter(experiment_variables, experiment_identifiers)
            )

            # group experiments by treatments
            treat_groups = experiment_rows.groupby(treatment_variables, dropna=False)
            for treatment_identifiers, treatment_rows in treat_groups:
                treatment = Treatment(
                    created_at=CREATED_AT,
                    experiment=experiment,
                    **treatment_setter(treatment_variables, treatment_identifiers)
                )

                # assign duplicate keys for repeated measurements
                if "replicate_id" not in treatment_rows.columns:
                    label_duplicates(treatment_rows, index=["time", "measurement"])
                
                # iterate over observations in treatment
                for _, row in treatment_rows.iterrows():
                    observation = Observation(
                        created_at=CREATED_AT,
                        experiment=experiment,
                        treatment=treatment,
                        measurement=row.measurement,
                        unit=row.unit,
                        replicate_id=row.replicate_id,
                        time=row.time,
                        value=row.value  
                    )

                    session.add(observation)

        session.flush()
        session.commit()


def remove_latest(database):
    engine = create_engine(f"sqlite:///{database}", echo=False)
    
    with Session(engine) as session:
        experiments = pd.read_sql(
            select(Experiment), 
            con=f"sqlite:///{database}"
        )

        created_last = experiments.created_at.unique()[-1]
        stmt = select(Experiment).where(Experiment.created_at == created_last)

        for row in session.execute(stmt):
            session.delete(row.Experiment)

        session.flush()
        session.commit()

def create_database(database):
    # SQLAlchemy does not suppor the addition of columns. This has to be done
    # by hand, but this is also not such a big deal. 
    engine = create_engine(f"sqlite:///{database}", echo=False)
    session = Session(engine)
    if not os.path.exists(database):
        Base.metadata.create_all(engine)

def delete_tables(database):
    engine = create_engine(f"sqlite:///{database}", echo=False)
    session = Session(engine)
    if os.path.exists(database):
        Base.metadata.drop_all(engine)


if __name__ == "__main__":
    database = "data/tox.db"
    create_database(database)
    add_data(
        database=database, 
        data=data, 
        experiment_variables=experiment_variables,
        treatment_variables=treatment_vars,
        experiment_setter=experiment_setter,
        treatment_setter=treatment_setter,
    )