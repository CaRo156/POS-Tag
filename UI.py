from tkinter import *
from tkinter import ttk
from HMM import HMM, get_train_test_data

class main_window():

    def __init__(self):
        X_train, Y_train, _, _ = get_train_test_data(split=0.01)
        self.hmm = HMM()
        self.hmm.train(X_train, Y_train)

        self.top = Tk()
        self.top.title('POS Tagging')
        self.top.geometry('500x350')
        self.process_btn = Button(self.top, text='Process', command=self.process_btn_click)
        self.process_btn.place(x=400, y=300)

        self.input_label = Label(self.top, text='Input:  ')
        self.input_label.place(x=20, y=10)
        self.input_text = Text(self.top, width=65, height=7)
        self.input_text.place(x=20, y=35)

        self.result_label = Label(self.top, text='Result:  ')
        self.result_label.place(x=20, y=150)
        self.result_text = Text(self.top, width=65, height=8)
        self.result_text.place(x=20, y=170)
        self.result_text.config(state=DISABLED)

        self.top.mainloop()

    def process_btn_click(self):
        content = self.input_text.get('1.0', END)
        content = content.replace('\n', '').replace(',', ' ,')
        if content.find('.') != -1:
            content = content.split('.')[:-1]
            result = []
            for l in content:
                l = l.strip() + ' .'
                tags = self.hmm.viterbi_bigram(l)
                result += [c + '/' + t for c, t in zip(l.split(' '), tags)]
        else:
            tags = self.hmm.viterbi_bigram(content)
            result = [c + '/' + t for c, t in zip(content.split(' '), tags)]
        self.result_text.config(state=NORMAL)
        self.result_text.delete('1.0', END)
        self.result_text.insert('1.0', result)
        self.result_text.config(state=DISABLED)

if __name__ == '__main__':
    mw = main_window()