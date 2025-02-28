import plotly.graph_objects as go
from skimage import measure

def plot_3d_segmentation(segmentation, threshold=0.5, save_path=None):
    """
    Plots a 3D visualization of the segmentation using Plotly.

    Args:
        segmentation (np.ndarray): Segmented mask.
        threshold (float): Threshold value for marching cubes.
        save_path (str, optional): If provided, saves the visualization as an HTML file.

    Returns:
        fig (plotly.graph_objects.Figure): The 3D plot.
    """
    verts, faces, normals, values = measure.marching_cubes(segmentation, level=threshold)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightpink',
            opacity=0.50
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Segmentation Visualization'
    )

    if save_path:
        fig.write_html(save_path)

    fig.show()
