from diffusers import DiffusionPipeline
import os

# Create folder .out if it doesn't exist
if not os.path.exists(".out"):
    os.mkdir(".out")

pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to("mps")

i = 0


def generate_image(prompt: str):
    global i

    results = pipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
    )
    image = results.images[0]
    image.save(f".out/image_{i}.png")
    # Write the prompt to a file
    with open(f".out/image_{i}_prompt.txt", "w") as f:
        f.write(prompt)
    i += 1
    return image


# Loop until the user enters exit or sends a SIGINT (Ctrl+C)
while True:
    prompt = input("Enter a prompt: ")
    if prompt == "exit":
        break
    generate_image(prompt)
