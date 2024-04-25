# Example with D17 data

- Use the `fit_steady_states.ipynb` notebook to fit the steady states.

    You have to define a starting structure. This can either be the initial
    or final state. If the final state is more complex and has more layers,
    it's generally best to start with that.

    The fit needs to be done with the `refl1d` reflectometry package so that
    we produce a `<model name>-expt.json` file. This file contains the structure that will be used as a starting point.

    The structure in this notebook is defined in the `create_model(...)` 
    function.


- Use the `RL-training.ipynb` notebook to train.

    When executing the model, you have to pick whether to fit chronologically or in reverse. If you want to start with the final state, use

    ```
        REVERSE = True
    ```

    You can also reload a trained model by settings
    ```
        TRAIN = False
    ```
    in the 6th cell.

- Use the `training_view.ipynb` to monitor the training while it is running.

    This notebook is very similar to the training notebook, but loads
    snapshots of the model as it is being trained. This way you can monitor
    if it continues to improve. 

    The bottom cells just load the last model and uses it.

# Notes and next steps

In this case we used one in every five curve, about a minute apart, except for the first 10 curves.
So we have 88 curves.
Although this is lower than the total number of available curves, it shows an interesting
limitation of the approach. There are two timescales involved. Fast changes happen in the first
few curves, and slow change happens afterwards. Because we maximize the total reward, the
disagreement at early times is outweighed by the good agreement at later times.
One could run the training longer and eventually have a better agreement at early times, but
this would not be efficient. Picking how many curves we skip also influences how fast we
converge.

A better approach would be to add an action parameter to choose a number of points/times to
skip. Once could approximate the evolution between two times as being linear and compute the
average $\chi^2$ for the times that were skipped. That would reduce the weight given to 
slow-changing times, while still using them to get feedback. The algorithm would then pick
how to sample the time series, picking more points close the fast changes.