# gui with a textbox to enter an instance which is then classified
# and the result is displayed in the textbox
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import ttk
import subprocess

path = ''

def generateGui():
    root = tk.Tk()

    root.title("Classifier")
    root.geometry("510x230")
    style = ttk.Style(root)
    # root.tk.call('source', 'Azure-ttk-theme/azure.tcl')
    # style.theme_use('azure-light')
    root.tk.call('source', 'Forest-ttk-theme/forest-light.tcl')
    style.theme_use('forest-light')

    root.resizable(False, False)
    # root.configure(background="white")

    textbox = scrolledtext.ScrolledText(root, wrap=None, width=60, height=10)
    textbox.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
    # textbox = tk.Text(root, height=10)
    # textbox.grid(row=0, column=0, sticky="ew")
    # scrollbar = ttk.Scrollbar(root, command=text.yview, orient="vertical")
    # scrollbar.grid(row=0, column=1, sticky="ns")
    # textbox.configure(yscrollcommand=scrollbar.set)

    button = ttk.Button(root, text="Open", command=lambda: openFile(textbox))
    button.grid(row=1, column=1, padx=5, pady=5)

    button = ttk.Button(root, text="Classify", command=lambda: classify(textbox))
    button.grid(row=1, column=0, padx=5, pady=5)

    root.mainloop()


def openFile(textbox):
    global path
    filename = filedialog.askopenfilename(title="Select file", filetypes=[("Text files", "*.txt"), ("all files", "*.*")])
    path = filename
    if filename == "":
        return
    try:
        with open(filename, "r") as f:
            textbox.delete("1.0", tk.END)
            textbox.insert(tk.END, f.read())
    except Exception as e:
        messagebox.showerror("Error", str(e))


def executeJar():
    # filename = './datuak/test.csv'
    subprocess.call(
        ['java', '-jar', './jar/testMiningWeka.jar', './modeloa/modeloaRandomForest.model', path,
         './modeloa/iragarpen.txt'
            , './Dictionary/hiztegia.txt'])


def classify(textbox):

    executeJar()
    text = textbox.get("1.0", tk.END)
    if text == "":
        messagebox.showinfo("Error", "No text entered")
        return
    try:
        with open('./modeloa/iragarpen.txt', "r") as f:
            textbox.delete("1.0", tk.END)
            textbox.insert(tk.END, f.read())

    except Exception as e:
        messagebox.showerror("Error", str(e))


generateGui()
