import glob as _glob
import os as _os


def get_path(
    dir_base_experiment: str = "/work/bm1124/m300086/CMIP6/experiments",
    member: int = 1,
    init: int = 1960,
    model: str = "hamocc",
    output_stream: str = "monitoring_ym",
    timestr: str = "*1231",
    ending: str = "nc",
) -> str:
    """Get the path of a file for MPI-ESM standard output file names and directory.

    Args:
        dir_base_experiment (str): Path of experiments folder. Defaults to
            "/work/bm1124/m300086/CMIP6/experiments".
        member (int): member label. Defaults to 1.
        init (int): initialization label. Typically year. Defaults to 1960.
        model (str): submodel name. Defaults to "hamocc".
            Allowed: ['echam6', 'jsbach', 'mpiom', 'hamocc'].
        output_stream (str): output_stream name. Defaults to "monitoring_ym".
            Allowed: ['data_2d_mm', 'data_3d_ym', 'BOT_mm', ...]
        timestr (str): timestr likely including *. Defaults to "*1231".
        ending (str): ending indicating file format. Defaults to "nc".
            Allowed: ['nc', 'grb'].

    Returns:
        str: path of requested file(s)

    """
    # get experiment_id
    dirs = _os.listdir(dir_base_experiment)
    experiment_id = [
        x for x in dirs if (f"{init}" in x and "r" + str(member) + "i" in x)
    ]
    assert len(experiment_id) == 1
    experiment_id = experiment_id[0]  # type: ignore
    dir_outdata = f"{dir_base_experiment}/{experiment_id}/outdata/{model}"
    path = f"{dir_outdata}/{experiment_id}_{model}_{output_stream}_{timestr}.{ending}"
    if _os.path.exists(_glob.glob(path)[0]):
        return path
    else:
        raise ValueError(f"Path not found or no access: {path}")
