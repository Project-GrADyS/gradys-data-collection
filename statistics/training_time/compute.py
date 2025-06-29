import os

import pandas as pd

for path in sorted(os.listdir("./data")):
    run_name = path[4:path.index("-tag")]
    df = pd.read_csv(os.path.join("./data", path))
    df["Wall time"] = pd.to_datetime(df["Wall time"], unit='s')
    start_time = df["Wall time"].min()

    # Time in which the "Value" column first exceeds 0.95
    succesfull_time = df[df["Value"] >= 0.95]["Wall time"].min()

    print(f"Run: {run_name}")
    print(f"Start time: {start_time}")
    print(f"Succesfull time: {succesfull_time}")
    print(f"Time to success (h): {(succesfull_time - start_time).total_seconds() / 3600:.2f}")
    print("----------------------")