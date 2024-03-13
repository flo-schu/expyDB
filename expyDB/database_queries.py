from sqlalchemy import select, or_
from sqlalchemy.orm import selectinload, Session, joinedload
from expyDB.database_model import Treatment, Observation, Experiment


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

