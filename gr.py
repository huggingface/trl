import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

import matplotlib.pyplot as plt

def plot_forecast(num_param, batch_size, precision,     seq_len):
    num_param = float(num_param)*1e9
    precision = {"float32": 4, "float16": 2, "bfloat16": 2}[precision]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    # Model Parameters: N×precision
    y1 = num_param*precision / (1024**3)

    # Optimizer States: 2×N×precision
    y2 = 2*num_param*precision / (1024**3)

    # Activations: B×Sequence Length×K×precision
    K = 4.6894e-04*num_param + 1.8494e+06
    y3 = batch_size*seq_len*K*precision / (1024**3)
    
    # Gradients: N×precision
    y4 = num_param*precision / (1024**3)

    # Create stacked bars
    ax.bar(0, y1, color='r')
    ax.bar(0, y2, bottom=y1, color='b')
    ax.bar(0, y3, bottom=y1+y2, color='g')
    ax.bar(0, y4, bottom=y1+y2+y3, color='y')

    # Add text labels inside the bars
    ax.text(0, y1 / 2, 'Model Parameters', ha='center', va='center', color='white', fontweight='bold')
    ax.text(0, y1 + y2 / 2, 'Optimizer States', ha='center', va='center', color='white', fontweight='bold')
    ax.text(0, y1 + y2 + y3 / 2, 'Activations', ha='center', va='center', color='white', fontweight='bold')
    ax.text(0, y1 + y2 + y3 + y4 / 2, 'Gradients', ha='center', va='center', color='white', fontweight='bold')

    # remove x axis
    ax.xaxis.set_visible(False)
    
    # Set GB as the unit for the y-axis
    ax.set_ylabel('Memory (GB)')
    fig.tight_layout()
    return fig


demo = gr.Interface(
    plot_forecast,
    [
        gr.Number(7, label="Number of parameters (B)"),
        gr.Radio([1, 2, 4, 8, 16, 32, 64, 128], value=8, label="Batch size"),
        gr.Radio(["float32", "float16", "bfloat16"],value="float32", label="Precision"),
        gr.Slider(1, 1024, label="Sequence Length", step=1, value=128),
    ],
    gr.Plot(label="forecast", format="png"),
)

if __name__ == "__main__":
    demo.launch()
