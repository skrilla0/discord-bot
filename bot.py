from discord import Intents
from discord.ext import commands
from dotenv import load_dotenv
import os
import replicate
import discord
from io import BytesIO
 
load_dotenv()
 
intents = Intents.default()
intents.message_content = True
 
bot = commands.Bot(
    command_prefix="!",
    description="Your Image Generator Bot",
    intents=intents,
)
 
 
@bot.command()
async def flux(ctx, *, prompt):
    """Generate an image from a text prompt using the Stable Diffusion 3.5 model"""
    try:
        msg = await ctx.send(f'"{prompt}"\n> Generating...')
 
        output = replicate.run(
            "stability-ai/stable-diffusion-3.5-large",
            input={
                "prompt": prompt,
                "cfg": 4.5,
                "steps": 40,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90,
                "prompt_strength": 0.85
            }
        )
 
        file_data = output[0].read()
        file = discord.File(BytesIO(file_data), filename="flux.png")
 
        await msg.delete()
        await ctx.send(f'"{prompt}"', file=file)
    
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")


@bot.command()
async def byte(ctx, *, prompt):
    """Generate an image from a text prompt using the SDXL-Lightning model"""
    try:
        msg = await ctx.send(f'"{prompt}"\n> Generating...')
 
        output = replicate.run(
            "bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",
            input={
                "width": 1024,
                "height": 1024,
                "prompt": prompt,
                "scheduler": "K_EULER",
                "num_outputs": 1,
                "guidance_scale": 0,
                "negative_prompt": "worst quality, low quality",
                "num_inference_steps": 4
            }
        )

        # Create embed for better presentation
        embed = discord.Embed(title="SDXL-Lightning Generation", color=discord.Color.blue())
        embed.add_field(name="Prompt", value=prompt, inline=False)
        embed.set_image(url=output[0])
        embed.set_footer(text=f"Requested by {ctx.author.name}")

        await msg.delete()
        await ctx.send(embed=embed)
    
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")
 
bot.run(os.environ["DISCORD_TOKEN"])