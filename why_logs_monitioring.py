import os
import whylogs as why
from score import run
import pandas as pd
from whylogs.viz import NotebookProfileVisualizer
import whylogs as why
import matplotlib.pyplot as plt

os.getcwd()
# def monitor(wine_reference, wine_target)


wine_reference = pd.read_csv("occupancy_data/training_data.txt")
wine_reference = wine_reference.drop(columns = ['date'])

def target_monitor(target):
    wine_target_data = target
    target_pred = run(wine_target_data, "occupancy_model_main.bin")
    wine_target = pd.read_csv(wine_target_data)
    wine_target["Occupancy"] = pd.Series(target_pred)
    wine_target = wine_target.drop(columns = ['Unnamed: 0'])

    return wine_target


# print(wine_target)
# print(wine_reference)

wine_target_4 = target_monitor("occupancy_data/occupancy_batch_3_.csv")
wine_target_5 = target_monitor("occupancy_data/occupancy_batch_2_.csv")
wine_target_6 = target_monitor("occupancy_data/occupancy_batch_1_.csv")
# wine_target_7 = target_monitor("occupancy_for_3days/occupancy_data_7_.csv")



result = why.log(pandas=wine_target_4)
profile = result.profile()

profile.track(pandas=wine_target_5)
# profile.track(pandas=wine_target_6)
# profile.track(pandas=wine_target_7)

prof_view = profile.view()

result_ref = why.log(pandas=wine_reference)
prof_view_ref = result_ref.view()

visualization = NotebookProfileVisualizer()
visualization.set_profiles(target_profile_view=prof_view, reference_profile_view=prof_view_ref)

profile_html = visualization.summary_drift_report()

visualization.write(profile_html , html_file_name=os.getcwd() + "/profile_4_5_n.bin")

# if __name__ == '__main__':
#     monitor()