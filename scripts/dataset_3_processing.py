import numpy as np
import pandas as pd
import xarray as xr
from timepath.helpers.misc import get_grouped_unique_val, replace_na

def cint():
    """Works along the lines of sublethal effects data import from dataset 2.
    No substance dimension
    """
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
    data["sample_id"] = (
        data["experiment_date"].astype(str) + "_" +
        np.where(data["mix"], "mix", data["substance"]) + "_" + 
        data["hpf"].astype(str) + "_" +
        data["hpe"].astype(str) + "_" +
        np.round((data["cext_nom"] * 1e6),2).astype(str)
    )

    # keep in mind that datapoints are not strongly connected with one another.
    # we assign a replicate id for each datapoint that was reported multiple times
    data["rep_no"] = 0
    for key, group in data.groupby(["sample_id", "substance"]):
        if len(group) == 1:
            continue
        for rep, (rkey, row) in enumerate(group.iterrows()):
            data.loc[rkey, "rep_no"] = rep

    data["id"] = (
        data["experiment_date"].astype(str) + "_" +
        np.where(data["mix"], "mix", data["substance"]) + "_" + 
        np.round((data["cext_nom"] * 1e6),2).astype(str) + "_" + 
        data["hpf"].astype(str) + "_" +
        data["rep_no"].astype(str)
    )

    data = data.drop(columns=["experiment_date","sample_id","n_zfe","rep_no","hpe"])
    mix_id = get_grouped_unique_val(data, "mix", "id")
    sub_id = get_grouped_unique_val(data, "substance", "id")
    hpf_id = get_grouped_unique_val(data, "hpf", "id")
    cext_id = get_grouped_unique_val(data, "cext_nom", "id")
    exp_id = get_grouped_unique_val(data, "experiment_id", "id")
    
    data = data.set_index(["time", "id"])
    # data = data.set_index(["time", "substance", "id"])
    ds = xr.Dataset.from_dataframe(data)

    ds = ds.assign_coords({
        "mix": ("id", mix_id),
        "substance": ("id", sub_id),
        "cext_nom": ("id", cext_id),
        "hpf": ("id", hpf_id),
        "experiment_id": ("id", exp_id)
    })
    
    
    ds.attrs["cext"] = "measured external concentration in mol L-1 based on cint_zfe and estimated volume of one zfe at time t"
    ds.attrs["cint"] = "calculated internal concentration in mol L-1"
    ds.attrs["cext_nom"] = "nominal external concentration in mol L-1"
    ds.attrs["cint_zfe"] = "calculated internal amount in one ZFE mol based on concentration in extract from exposed ZFES (n=n_zfe). Data in raw files"

    ds.to_netcdf("data/processed_data/ds3_cint.nc")


def cint_to_database():

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

if __name__ == "__main__":
    cint()