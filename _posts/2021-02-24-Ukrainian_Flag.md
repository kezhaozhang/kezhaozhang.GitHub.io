---
title: In Support of Ukraine: Flag of Ukraine with Matplotlib
typora-root-url: ../../kezhaozhang.GitHub.io
---

In support of Ukraine and the Ukrainian people, I made a Ukrainian flag with matplotlib.

<figure class="image">
  <center>
    <img src='/assets/images/ua.svg' height="500">
  </center>
</figure>

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()
ax.plot()
ax.add_patch(Rectangle((0,0), 3, 1, color='#ffd700'))
ax.add_patch(Rectangle((0,1), 3, 1, color='#0057b7'))
ax.set_axis_off();
```



