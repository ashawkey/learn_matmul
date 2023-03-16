import json
import matplotlib.pyplot as plt
import glob

# find all logs and plot them
for log in glob.glob("./logs/*.json"):
    with open(log, "r") as f:
        res = json.load(f)
    xs = [int(x) for x in res.keys()]
    ys = [res[str(x)]["gflops"] for x in xs]
    plt.plot(xs, ys, 'o-', label=log[7:-5])

plt.legend()
plt.xlabel("scale")
plt.ylabel("GFLOPs")
plt.show()