import pandas as pd
import xarray as xr
from sqlalchemy import select, create_engine, inspect, or_, and_
from sqlalchemy.orm import selectinload, Session, joinedload
from expyDB.database_model import Treatment, Observation, Experiment

from pymob.utils.misc import get_grouped_unique_val

stmt = select(Observation)
stmt = select(Treatment).options(selectinload(Treatment.observations)).order_by(Treatment.id)

stmt = (
    select(Observation, Treatment)
    .join(Treatment.observations)
    .where(Treatment.cext_nom_diuron > 0)
    .where(Treatment.cext_nom_diclofenac == 0)
    .where(Treatment.cext_nom_naproxen == 0)
    .where(or_(
        Observation.measurement == "cint_diuron",
        Observation.measurement == "cext_diuron",
    ))
    .order_by(Observation.id)
)


def query(database, statement):
    """ask a query to the database.

    Parameters
    ----------

    database[str]: path to a database
    statement[Select]: SQLAlchemy select statement
    """
    # ask query to database
    data = pd.read_sql(statement, con=f"sqlite:///{database}")
    
    data = data.drop(columns=["id_1", "experiment_id_1"])
    data["id"] = data["treatment_id"].astype(str) + "_" + data["replicate_id"].astype(str)

    data = data.set_index(["time", "id"])
    hpf_id = get_grouped_unique_val(data, "hpf", "id")
    cext_diu_id = get_grouped_unique_val(data, "cext_nom_diuron", "id")
    cext_dic_id = get_grouped_unique_val(data, "cext_nom_diclofenac", "id")
    cext_nap_id = get_grouped_unique_val(data, "cext_nom_naproxen", "id")
    nzfe_id = get_grouped_unique_val(data, "nzfe", "id")
    treat_id = get_grouped_unique_val(data, "treatment_id", "id")
    experi_id = get_grouped_unique_val(data, "experiment_id", "id")

    meta = (data["measurement"] + "___" + data["unit"]).unique()
    meta = {measure:unit for measure, unit in [m.split("___") for m in meta]}

    data = data.drop(columns=[
        "name", "experiment_id", "replicate_id", "unit", "nzfe", 
        "hpf", "treatment_id"
    ])

    data = data.pivot(columns="measurement", values="value")

    ds = xr.Dataset.from_dataframe(data)
    ds = ds.assign_coords({
        "cext_nom_naproxen": ("id", cext_nap_id),
        "cext_nom_diclofenac": ("id", cext_dic_id),
        "cext_nom_diuron": ("id", cext_diu_id),
        "hpf": ("id", hpf_id),
        "nzfe": ("id", nzfe_id),
        "treatment_id": ("id", treat_id),
        "experiment_id": ("id", experi_id),

    })

    ds.attrs.update(meta)

    return ds