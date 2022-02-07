import csv
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np

without_amnioml: List[float] = [0.0, 0.0, 0.0]
with_amnioml: List[float] = [0.0, 0.0, 0.0]


cm = 1 / 2.54  # centimeters in inches
w = 8.5
h = 2.5 * w / 4

plt.rc("font", size=7)
plt.rc("axes", labelsize=8)
fig, ax = plt.subplots(figsize=(w * cm, h * cm))


with open("data/csv/amnioml-reviews-times.csv", newline="") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=";")
    for row in spamreader:
        rating = int(row[1])
        time_without_amnioml = float(row[-2]) * 60
        time_with_amnioml = float(row[-1]) * 60

        if rating == 5:
            # use runtime,
            time_with_amnioml = float(row[-3].split(" ")[0])

        without_amnioml[rating - 1 - 2] += time_without_amnioml
        with_amnioml[rating - 1 - 2] += time_with_amnioml

with_amnioml[-1] = round(with_amnioml[-1])
labels = ["3", "4", "5"]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, without_amnioml, width, label="Without AmnioML")
rects2 = ax.bar(x + width / 2, with_amnioml, width, label="With AmnioML")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Time (seconds)")
ax.set_xlabel("Rating")
# ax.set_title('Total Segmentation Times by Rating')
# ax.set_xticks(x, labels)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.ylim(0, 8000)
plt.yticks([0, 2000, 4000, 6000, 8000])

# plt.save("total_segmentation_times_by_rating.pgf")

plt.savefig(sys.stdout.buffer, bbox_inches="tight", format="pgf")
