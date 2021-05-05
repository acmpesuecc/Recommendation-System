import tkinter as tk
import techwolfblitz1
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 300)
canvas1.pack()

canvas2 = tk.Canvas(root, width = 400, height = 300)
canvas2.pack()

entry1 = tk.Entry (root) 
entry2 = tk.Entry (root) 

canvas1.create_window(200, 140, window=entry1)
canvas2.create_window(200, 140, window=entry2)

button1 = tk.Button(text='Enter',)
canvas1.create_window(200, 180, window=button1)
button2 = tk.Button(text='Enter',)
canvas2.create_window(200, 180, window=button2)

root.mainloop()

if __name__ == '__main__':
  get_recommendation(entry1,entry2)