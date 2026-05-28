
import gradio as gr
from agent.agent import query_agent

def solar_chat(message, history):
    try:
        response = query_agent(message)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Define example questions
examples = [
    "What is the annual solar potential in Sydney?",
    "Which city is better for a solar farm: Melbourne or Brisbane?",
    "If I build a 5000 m² solar farm in Perth, what is the estimated daily yield?",
    "Estimate the solar potential for a site with 9 hours of sunshine and 30°C average temperature.",
    "How does average cloud cover in Darwin affect its long-term solar output?"
]

demo = gr.ChatInterface(
    fn=solar_chat,
    title="Australia's Solar Output Prediction System",
    description="Ask me about solar energy production in Australia based on weather conditions and city location.",
    examples=examples
)

if __name__ == "__main__":
    demo.launch(share=False) # No publicly shareable link
