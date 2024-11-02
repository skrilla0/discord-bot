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
    """Generate an image from a text prompt using the Faster, better FLUX Pro. Text-to-image model with excellent image quality, prompt adherence, and output diversity."""
    try:
        msg = await ctx.send(f'"{prompt}"\n> Generating...')
 
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input={
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 80,
                "safety_tolerance": 2,
                "prompt_upsampling": True
            }
        )
 
        file_data = output[0].read()
        file = discord.File(BytesIO(file_data), filename="flux.png")
 
        await msg.delete()
        await ctx.send(f'"{prompt}"', file=file)
    
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")
 
bot.run(os.environ["DISCORD_TOKEN"])