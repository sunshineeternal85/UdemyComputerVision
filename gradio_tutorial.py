# %%

import gradio as gr
import numpy as np
from typing import List

# %%

def greet1(name: str):
    return 'hello ' + name + '!'

def greet2(name: str, is_morning: bool, temperature: float):
    salutation = 'hello ' if is_morning else 'Good devening'
    greeting = f'{salutation} {name}. It is {temperature} degrees today'
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius,2)


def greet3(first_name, last_name):
    greet =  f'hello {first_name} {last_name}'
    return greet


def greet3(first_name, last_name, street, postcode, city, country):
    greet =  (f"hello {first_name} {last_name} \n"
              f"you are living on {street} {postcode}, \n"
              f"in {city} {country} \n"
              )
    
    return greet

def calc(number: str):
    number = int(number)
    return f'Squared of : {number} =  {number**2}'


def calc2(number: str):
    number = int(number)
    return f'Squared of : {number} =  {number**2}'

def sepia(input_image: np.array):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])
    sepia_img = input_image.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    print(input_image.shape, sepia_img.shape)
    return sepia_img





# %%
if __name__ == '__main__':
    if False:
        demo = gr.Interface(fn=greet1,
                            inputs=gr.Textbox(lines=2, 
                                              placeholder='Number here')
                            , outputs='text')
        demo.launch()


    if False:
        demo2 = gr.Interface(fn=greet2, 
                            inputs=['text', 'checkbox', gr.Slider(0,100)],
                            outputs=['text', 'number'])
        demo2.launch()

    if False:
        image = np.random.randint(0,255,(3,3))
        
        demo = gr.Interface(
            fn=sepia,
            inputs=gr.Image( type='numpy'),
            outputs='image'
        )
        demo.launch()


    if False:
        with gr.Blocks() as demo:
            first_name = gr.Textbox(label='First name')
            last_name = gr.Textbox(label='Last Name')
            output = gr.Textbox(label='output box')
            greet_btn = gr.Button('Greet')
            greet_btn.click(fn=greet3, inputs=[first_name, last_name], outputs=output)
        demo.launch()

    if True:
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    first_name = gr.Textbox(label='First name')
                    last_name = gr.Textbox(label='Last Name')

                with gr.Column():
                    street = gr.Textbox(label='Street')
                    post_code = gr.Textbox(label='Post Code')
                    city = gr.Textbox(label='City')
                    country = gr.Textbox(label='Country')
            with gr.Row():
                with gr.Column():   
                        output = gr.Textbox(label= 'introducing...') 
                        greet_btn = gr.Button('Greet')
                        greet_btn.click(fn=greet3, 
                                        inputs=[
                                            first_name, 
                                            last_name,
                                            street,
                                            post_code,
                                            city,
                                            country], 
                                        outputs=output)
        demo.launch()


# %%
