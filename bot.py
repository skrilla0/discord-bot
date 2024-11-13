from discord import Intents
from discord.ext import commands
from dotenv import load_dotenv
import os
import replicate
import discord
from io import BytesIO
from openai import OpenAI
import anthropic
from groq import Groq
 
load_dotenv()
 
intents = Intents.default()
intents.message_content = True
 
bot = commands.Bot(
    command_prefix="!",
    description="Your Image Generator Bot",
    intents=intents,
)
 
# Stable Diffusion 3.5 Bot
@bot.command()
async def stab(ctx, *, prompt):
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

# bytedance SDXL-Lightning Bot
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
 

# xAI Grok Bot 
@bot.command()
async def grok(ctx, *, prompt):
    """Generate a response using xAI's Grok model"""
    try:
        msg = await ctx.send(f'"{prompt}"\n> Thinking...')
        
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

        completion = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
                {"role": "user", "content": prompt},
            ],
        )

        response = completion.choices[0].message.content

        # Create embed for better presentation
        embed = discord.Embed(
            title="Grok's Response", 
            description=response,
            color=discord.Color.purple()
        )
        embed.set_footer(text=f"Requested by {ctx.author.name}")

        await msg.delete()
        await ctx.send(embed=embed)
    
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")

# Anthropic Claude API 

@bot.command()
async def claude(ctx, *, prompt):
    """Generate a response using Anthropic's Claude model"""
    try:
        msg = await ctx.send(f'"{prompt}"\n> Thinking...')
        
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response = message.content[0].text

        # Create embed for better presentation
        embed = discord.Embed(
            title="Claude's Response", 
            description=response,
            color=discord.Color.green()
        )
        embed.set_footer(text=f"Requested by {ctx.author.name}")

        await msg.delete()
        await ctx.send(embed=embed)
    
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")

# ChatGPT

@bot.command()
async def chatgpt(ctx, *, prompt):
    """Generate a response using OpenAI's GPT-4 model"""
    try:
        msg = await ctx.send(f'"{prompt}"\n> Thinking...')
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for a cheaper/faster option
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )

        response = chat_completion.choices[0].message.content

        # Create embed for better presentation
        embed = discord.Embed(
            title="ChatGPT's Response", 
            description=response,
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Requested by {ctx.author.name}")

        await msg.delete()
        await ctx.send(embed=embed)
    
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")

# LLama Groq 70B 8192 tool use preview LLM via Groq's API - https://huggingface.co/Groq/Llama-3-Groq-70B-Tool-Use

@bot.command()
async def llama(ctx, *, prompt):
    """Generate a response using Groq's LLM models"""
    try:
        msg = await ctx.send(f'"{prompt}"\n> Thinking...')
        
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-groq-70b-8192-tool-use-preview",  # You can change the model as needed
            stream=False,
        )

        response = chat_completion.choices[0].message.content

        # Create embed for better presentation
        embed = discord.Embed(
            title="Groq's Response", 
            description=response,
            color=discord.Color.orange()
        )
        embed.set_footer(text=f"Requested by {ctx.author.name}")

        await msg.delete()
        await ctx.send(embed=embed)
    
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")

bot.run(os.environ["DISCORD_TOKEN"])