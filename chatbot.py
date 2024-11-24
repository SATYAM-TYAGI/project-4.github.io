import torch
import tkinter as tk
from tkinter import scrolledtext, Canvas
from PIL import Image, ImageTk
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ChatBot")
        self.root.geometry("600x600")
        
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.bg_image = ImageTk.PhotoImage(Image.open("background.jpg"))
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")
    
        self.chat_log = scrolledtext.ScrolledText(self.root, bg="lightyellow", font=("Helvetica", 12), wrap=tk.WORD)
        self.chat_log.place(relwidth=0.9, relheight=0.7, relx=0.05, rely=0.05)
        self.chat_log.config(state=tk.DISABLED)
        
        self.entry_box = tk.Entry(self.root, font=("Helvetica", 12))
        self.entry_box.place(relwidth=0.74, relheight=0.06, relx=0.05, rely=0.85)
        
        self.send_button = tk.Button(self.root, text="Send", font=("Helvetica", 12, "bold"), bg="lightblue", command=self.send_message)
        self.send_button.place(relwidth=0.15, relheight=0.06, relx=0.81, rely=0.85)
        
        self.entry_box.bind("<Return>", self.send_message)
    
    def send_message(self, event=None):
        user_input = self.entry_box.get()
        self.entry_box.delete(0, tk.END)
        if user_input.lower() == "quit":
            self.root.quit()
        else:
            self.display_message("You: " + user_input, "right")
            response = self.get_ai_response(user_input)
            self.display_message("Bot: " + response, "left")
    
    def get_ai_response(self, user_input):
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = new_user_input_ids if not hasattr(self, 'chat_history_ids') else torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
        
        self.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones(bot_input_ids.shape, device=bot_input_ids.device))

        response = tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

    def display_message(self, message, side):
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, "\n")
        
        cloud_box = Canvas(self.chat_log, bg="lightblue", highlightthickness=0, bd=0)
        cloud_box.create_text(10, 10, text=message, anchor="nw", width=400, font=("Helvetica", 12), fill="black")
        cloud_box.config(width=cloud_box.bbox("all")[2] + 20, height=cloud_box.bbox("all")[3] + 20)
        
        self.chat_log.window_create(tk.END, window=cloud_box, padx=5, pady=5)
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatBotGUI(root)
    root.mainloop()
