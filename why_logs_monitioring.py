import whylogs as why
from score import run
import pandas as pd
from whylogs.viz import NotebookProfileVisualizer
import whylogs as why
import matplotlib.pyplot as plt

# def monitor(wine_reference, wine_target)

wine_target = pd.read_csv("occupancy_for_3days/occupancy_data_2.csv")
wine_reference_ = "occupancy_for_3days/occupancy_data_4_.csv"

target_pred = run(wine_reference_, "occupancy_model.bin")
wine_reference = pd.read_csv(wine_reference_)
wine_reference["Occupancy"] = pd.Series(target_pred)
# wine_reference[wine_reference["Occupancy"] == 1]

fig = plt.figure()

result = why.log(pandas=wine_target)
prof_view = result.view()

result_ref = why.log(pandas=wine_reference)
prof_view_ref = result_ref.view()

visualization = NotebookProfileVisualizer()
visualization.set_profiles(target_profile_view=prof_view, reference_profile_view=prof_view_ref)

profile = visualization.profile_summary()

# why.write(profile,"profile.bin")

# if __name__ == '__main__':
#     monitor()