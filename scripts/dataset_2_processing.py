import numpy as np
import pandas as pd
import xarray as xr
from timepath.helpers.misc import get_grouped_unique_val

# WARNING: ids of Experiments do not match!!!

experiment_map = {
    30: {"substance": "Diuron", "hpf":0, "replicate": 1},
    42: {"substance": "Diuron", "hpf":24, "replicate": 1},
    46: {"substance": "Diuron", "hpf":24, "replicate": 2},
    61: {"substance": "Mix", "hpf":0, "replicate": 1},
    82: {"substance": "Mix", "hpf":24, "replicate": 1},
    83: {"substance": "Mix", "hpf":0, "replicate": 2},
    84: {"substance": "Mix", "hpf":24, "replicate": 2},
    85: {"substance": "Naproxen", "hpf":0, "replicate": 1},
    115: {"substance": "Naproxen", "hpf":24, "replicate": 2},
    119: {"substance": "Diclofenac", "hpf":0, "replicate": 1},
    124: {"substance": "Diclofenac", "hpf":24, "replicate": 2},
    219: {"substance": "Diuron", "hpf":0, "replicate": 2},
    245: {"substance": "Diuron", "hpf":24, "replicate": 3},
}

INDEX = ["time", "id"]

def mixture_tktd():
    meta = pd.read_csv("data/dataset_2/TKTD_Data_three compounds and mixture_112022_METADATA.csv")
    data = pd.read_csv("data/dataset_2/TKTD_Data_three compounds and mixture_112022.csv")
    data["time"] = data.hpf + data.hpe
    data["mix"] = np.where(data.mix == "y", True, False)
    data["id"] = data.hpf.astype(str) + "hpf_" + np.where(data.mix, "mix", data.substance) + "_rep" + data.rep_no.astype(str)
    
    mix_id = get_grouped_unique_val(data, "mix", "id")
    hpf_id = get_grouped_unique_val(data, "hpf", "id")

    data = data.set_index(["time", "substance", "id"])
    data = data.drop(columns=["rep_no", "hpe"])


    ds = xr.Dataset.from_dataframe(data)
    for i in range(12):
        name = meta.loc[i, "Column names"]
        desc = meta.loc[i, "Description"]

        ds.attrs[name] = desc

    general_remarks = ";\n".join([f"{line}" for line in meta.iloc[14:, 0]])

    ds.attrs["general remarks"] = general_remarks

    ds["mix"] = ("id", mix_id)
    ds["hpf"] = ("id", hpf_id)

    ds.to_netcdf("data/processed_data/ds2_tktd.nc")


def sublethal_effects():
    processed = pd.read_csv("data/dataset_2/observations_phenotypes_sorted_update.csv")

    mix_proportions = ("Diuron", 0.11, "Diclofenac", 0.026,"Naproxen", 0.864,)

    data = processed.rename(columns={
        "age_embryo_observation_hpf": "time",
        "age_embryo_exposure_start_hpf": "hpf",
        "test_substance": "substance",
        "concentration": "cext"
    }).drop(columns=[
        "observation_hpe", "effect.ratio", "concentration_ID", "unit",
        "cat_short", "parent_short", "effect_short", "position_x"
    ])

    for key, group in data.groupby("experiment_id"):
        data.loc[group.index, "substance"] = experiment_map[key]["substance"]

    data["mix"] = np.where(data.substance == "Mix", True, False)
    data["id"] = (
        "hpf:" + data.hpf.astype(str) 
        + "_substance:" + data.substance
        + "_expid:" + data.experiment_id.astype(str)
        + "_cext:" + data.cext.astype(str)
        + "_pos:" + data.container.astype(str) + "." + data.position_y.astype(str)
    )
    data["values"] = 1
    reg = "\s\/\s|\s|\/"
    data["detailed_effect"] = data.detailed_effect.str.replace(reg, "_", regex=True)
    data["parent_effect"] = data.parent_effect.str.replace(reg, "_", regex=True)
    data["effect_category"] = data.effect_category.str.replace(reg, "_", regex=True)

    detailed_effects = data.pivot_table(
        index=INDEX, columns="detailed_effect", values="values", fill_value=0)
    
    parent_effects = data.pivot_table(
        index=INDEX, columns="parent_effect", values="values", fill_value=0)
    
    category_effects = data.pivot_table(
        index=INDEX, columns="effect_category", values="values", fill_value=0)

    data = data.set_index(INDEX)
    data_no_duplicates = data.loc[~data.index.duplicated(keep="last")]

    mix_id = get_grouped_unique_val(data_no_duplicates, "mix", "id")
    hpf_id = get_grouped_unique_val(data_no_duplicates, "hpf", "id")
    subst_id = get_grouped_unique_val(data_no_duplicates, "substance", "id")
    cext_id = get_grouped_unique_val(data_no_duplicates, "cext", "id")
    exp_id = get_grouped_unique_val(data_no_duplicates, "experiment_id", "id")
    cont_id = get_grouped_unique_val(data_no_duplicates, "container", "id")
    posy_id = get_grouped_unique_val(data_no_duplicates, "position_y", "id")


    ds = xr.Dataset.from_dataframe(detailed_effects)
    ds[parent_effects.columns] = xr.Dataset.from_dataframe(parent_effects)
    ds[category_effects.columns] = xr.Dataset.from_dataframe(category_effects)

    ds = ds.assign_coords({
        "mix": ("id", mix_id),
        "substance": ("id", subst_id),
        "cext": ("id", cext_id),
        "hpf": ("id", hpf_id),
        "experiment_id": ("id", exp_id),
        "container": ("id", cont_id),
        "position_y": ("id", posy_id),
    })
    ds.attrs["substance_concentration"] = "µmol L-1"
    ds.attrs["mix_proportions"] = mix_proportions
    ds.attrs["detailed_effects"] = list(detailed_effects.columns)
    ds.attrs["parent_effects"] = list(parent_effects.columns)
    ds.attrs["category_effects"] = list(category_effects.columns)
    ds.to_netcdf("data/processed_data/ds2_sublethal_effects.nc")

if __name__ == "__main__":
    mixture_tktd()
    sublethal_effects()