import torch
from diffusers import AutoPipelineForText2Image

print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    variant="fp16",
    torch_dtype=torch.float32,
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres generar: ")

print("Generando imagen...")

imagen = modelo(
    prompt=prompt,
    num_inference_steps=1,
    guidance_scale=0.0,
).images[0]

imagen.save("imagen.png")
print("Imagen guardada como imagen.png")
