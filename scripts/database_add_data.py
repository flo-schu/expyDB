import warnings
import os
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
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

treatment_vars = ["substance", "mix", "cext_nom", "cext_nom_total", "hpf", "n_zfe"]

nans = data[treatment_vars].isna().values.sum(axis=0)

if np.any(nans > 0):
    warnings.warn(
        f"NaNs in treatment variables {np.array(treatment_vars)[nans > 0]} detected. " 
        "Fix in data input, define default, or live with nans in treatment info."
    )

# Create an engine to connect to your database
# SQLAlchemy does not suppor the addition of columns. This has to be done
# by hand, but this is also not such a big deal. 
database = "data/tox.db"

engine = create_engine(f"sqlite:///{database}", echo=False)
session = Session(engine)
if os.path.exists(database):
    Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)



with Session(engine) as session:
    # group by experiment
    exp_groups = data.groupby(["experiment_date", "experimentator"], dropna=False)
    for (exp_date, experimentator), experiment_rows in exp_groups:
        experiment = Experiment(
            date=exp_date,
            experimentator=experimentator
        )
        treat_groups = experiment_rows.groupby(treatment_vars,dropna=False)

        # group experiments by treatments
        for treatment_vals, treatment_rows in treat_groups:
            # TODO: Future this could be modified so that a treatment specific
            # function is provided to modify the information, returning a dict
            # with the kwargs of Treatment(**kwargs)
            
            (substance, mix, cext_nom, cext_nom_tot, hpf, n_zfe) = treatment_vals
            
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
            # print(n_zfe)
            treatment = Treatment(
                experiment=experiment,
                hpf=hpf,
                nzfe=n_zfe,
                **exposure_map,
            )

            # iterate over measurement variables (when data are organized not in
            # long format)
            label_duplicates(treatment_rows, index=["time"])
            # iterate over observations in treatment
            for key, row in treatment_rows.iterrows():
                observation = Observation(
                    experiment=experiment,
                    treatment=treatment,
                    measurement=row.measurement,
                    unit=row.unit,
                    replicate_id=row.rep_no,
                    time=row.time,
                    value=row.value  
                )

                session.add(observation)

    session.flush()
    session.commit()