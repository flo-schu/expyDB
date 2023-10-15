import os
import warnings
import inspect
from typing import List
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from expyDB.database_model import Observation, Treatment, Experiment, Base

def label_duplicates(data, index: List[str], duplicate_column="replicate_id"):
    data[duplicate_column] = 0
    for _, group in data.groupby(index):
        if len(group) == 1:
            continue
        for rep, (rkey, _) in enumerate(group.iterrows()):
            data.loc[rkey, duplicate_column] = rep


def setter(variables, identifiers):
    return {key:value for key, value in zip(variables, identifiers)}

def add_data(
    database: str,
    data: pd.DataFrame,
    treatment: callable,
    experiment: callable,
    observation: callable,
):
    """Add data to a database.
    
    The data must be a pandas.DataFrame instance and be of 'long form',
    meaning that every measurement must have a separate row.
    
    It is advisable that treatment columns do not contain NaN values 
    
    In order to populate a database, functions that return the objects (instances)
    of the database model need to be provided.

    The function arguments of these functions **must** be names that are present 
    in the provided DataFrame. They are read from the function and provide the
    grouping in the nested loop, which populates the database with a structured
    relationship model.

    For different databases the loop below would have to be adapted to accomodate
    the respective structure of the database, but the basic principle remains the
    same.

    This approach offers the user the flexibility to transform the input data
    on a row basis, at the data import stage without having to deal with the 
    intricacies of the underlying database model.

    Arguments
    ---------

    database: [str] A string to specify the sqlite database.
    data: [pd.DataFrame] A pandas DataFrame. Needs to be in long format
    experiment: [callable] function, which returns an Experiment object.
    experiment: [callable] function, which returns a Treatment object. 
    observation: [callable] function, which retruns an Observation object.

    The function arguments of the callables **must** be names that are present 
    in the provided DataFrame.
    """

    experiment_variables: List[str] = inspect.getfullargspec(experiment)[0]
    treatment_variables: List[str] = inspect.getfullargspec(treatment)[0]
    observation_variables: List[str] = inspect.getfullargspec(observation)[0]


    nans = data[treatment_variables].isna().values.sum(axis=0)
    if np.any(nans > 0):
        warnings.warn(
            f"NaNs in treatment variables {np.array(treatment_variables)[nans > 0]} detected. " 
            "Fix in data input, define default, or live with nans in treatment info."
        )

    # Create an engine to connect to your database
    CREATED_AT = datetime.datetime.now()
    engine = create_engine(f"sqlite:///{database}", echo=False)

    with Session(engine) as session:

        # group by experiment
        exp_groups = data.groupby(experiment_variables, dropna=False)
        for experiment_identifiers, experiment_rows in exp_groups:
            
            # create Experiment object from the user provided function
            experiment_obj: Experiment = experiment(*experiment_identifiers)
            experiment_obj.created_at=CREATED_AT

            # group experiments by treatments
            treat_groups = experiment_rows.groupby(treatment_variables, dropna=False)
            for treatment_identifiers, treatment_rows in treat_groups:

                # create Treatment object from the user provided function
                treatment_obj: Treatment = treatment(*treatment_identifiers)
                experiment_obj.treatments.append(treatment_obj)

                # assign duplicate keys for repeated measurements
                if "replicate_id" not in treatment_rows.columns:
                    label_duplicates(treatment_rows, index=["time", "measurement"])
                
                # iterate over observations in treatment
                for _, row in treatment_rows.iterrows():
                    observation_args = row[observation_variables].to_list()

                    # create Observation object from the user provided function
                    observation_obj: Observation = observation(*observation_args)
                    if "replicate_id" not in observation_variables:
                        observation_obj.replicate_id = int(row["replicate_id"])
                    treatment_obj.observations.append(observation_obj)
                    experiment_obj.observations.append(observation_obj)
            
            session.add(experiment_obj)

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


