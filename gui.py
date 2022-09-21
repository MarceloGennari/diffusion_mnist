"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com
"""
import PySimpleGUI as sg
import torch
from PIL import Image
import numpy as np
import io

from models import ConditionalUNet, DDPMConditionalSampler


if __name__ == "__main__":
    # Define the window's contents
    file_list_column = [
        [sg.Button(" "), sg.Button("0"), sg.Button(" ")],
        [sg.Button("1"), sg.Button("2"), sg.Button("3")],
        [sg.Button("4"), sg.Button("5"), sg.Button("6")],
        [sg.Button("7"), sg.Button("8"), sg.Button("9")],
    ]

    image_viewer_column = [
        [sg.Text(f"Timestamp: - \t Label: -", key="-Title-")],
        [sg.Image(key="-IMAGE-", size=(256, 256))],
    ]

    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    # Create the window
    window = sg.Window("Window Title", layout)

    model = ConditionalUNet()
    model.load_state_dict(torch.load('unet_mnist.pth'))

    model.eval()
    sampler = DDPMConditionalSampler(model)

    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED or event == "Quit":
            break
        # Output a message to the window
        if event.isdigit() and int(event) in [x for x in range(10)]:
            with torch.no_grad():
                label = int(event)
                xt = torch.randn((1, 1, 28, 28))
                for t in range(999, -1, -1):
                    xt = sampler(xt, t, label)
                    if t % 10 == 0:
                        myarray = xt[0].cpu().numpy()
                        myarray = np.transpose(myarray, (1, 2, 0))

                        min_value = np.amin(myarray)
                        max_value = np.amax(myarray)

                        myarray = (myarray - min_value) / (max_value - min_value)
                        myarray = np.repeat(myarray, 3, axis=2)

                        myarray = np.uint8(myarray * 255)
                        im = Image.fromarray(myarray)

                        im = im.resize((256, 256), resample=Image.Resampling.NEAREST)
                        with io.BytesIO() as image:
                            im.save(image, format="png")
                            window["-IMAGE-"].update(data=image.getvalue())

                        window["-Title-"].update(f"Timestamp: {t} \t Label: {label}")
                        window.refresh()

    # Finish up by removing from the screen
    window.close()
