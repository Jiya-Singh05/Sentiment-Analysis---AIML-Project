import tkinter as tk
from tkinter import ttk, font as tkfont
import re
import threading

import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

_vader = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def get_vader_sentiment(text: str):
    scores = _vader.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, round(compound, 4)

def get_textblob_sentiment(text: str):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        label = 'Positive'
    elif polarity < 0:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, round(polarity, 4)

PALETTE = {
    'bg':        '#0F1117',
    'card':      '#1A1D27',
    'border':    '#2A2D3E',
    'accent':    '#6C63FF',
    'accent2':   '#FF6584',
    'text':      '#E8E9F3',
    'subtext':   '#8B8FA8',
    'positive':  '#2ECC71',
    'negative':  '#E74C3C',
    'neutral':   '#F39C12',
    'btn_hover': '#5A52D5',
}

LABEL_COLOR = {
    'Positive': PALETTE['positive'],
    'Negative': PALETTE['negative'],
    'Neutral':  PALETTE['neutral'],
}

class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Sentiment Analyzer')
        self.configure(bg=PALETTE['bg'])
        self.resizable(True, True)
        self.minsize(640, 640)

        self._load_fonts()
        self._build_ui()

        self.update_idletasks()
        w, h = 760, 820
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f'{w}x{h}+{x}+{y}')

    def _load_fonts(self):
        self.font_title   = tkfont.Font(family='Georgia',       size=22, weight='bold')
        self.font_heading = tkfont.Font(family='Georgia',       size=13, weight='bold')
        self.font_body    = tkfont.Font(family='Courier New',   size=11)
        self.font_mono    = tkfont.Font(family='Courier New',   size=12)
        self.font_label   = tkfont.Font(family='Helvetica',     size=10)
        self.font_score   = tkfont.Font(family='Helvetica',     size=28, weight='bold')
        self.font_badge   = tkfont.Font(family='Helvetica',     size=14, weight='bold')
        self.font_btn     = tkfont.Font(family='Helvetica',     size=12, weight='bold')
        self.font_sub     = tkfont.Font(family='Helvetica',     size=9)

    def _build_ui(self):
        root_frame = tk.Frame(self, bg=PALETTE['bg'])
        root_frame.pack(fill='both', expand=True, padx=28, pady=24)

        hdr = tk.Frame(root_frame, bg=PALETTE['bg'])
        hdr.pack(fill='x', pady=(0, 20))

        tk.Label(hdr, text='✦  Sentiment Analyzer',
                 font=self.font_title,
                 bg=PALETTE['bg'], fg=PALETTE['text']).pack(side='left')

        tk.Label(hdr, text='VADER  ·  TextBlob',
                 font=self.font_sub,
                 bg=PALETTE['bg'], fg=PALETTE['subtext']).pack(side='right', pady=(8, 0))

        self._divider(root_frame)

        in_card = self._card(root_frame, fill='x', pady=(14, 0))

        tk.Label(in_card, text='Quick Input',
                 font=self.font_label,
                 bg=PALETTE['card'], fg=PALETTE['subtext']).pack(anchor='w', pady=(0, 4))

        entry_frame = tk.Frame(in_card, bg=PALETTE['border'],
                               highlightbackground=PALETTE['border'],
                               highlightthickness=1, bd=0)
        entry_frame.pack(fill='x', pady=(0, 12))

        self.entry_box = tk.Entry(
            entry_frame,
            font=self.font_body,
            bg='#12141F',
            fg=PALETTE['text'],
            insertbackground=PALETTE['accent'],
            relief='flat',
            bd=0,
            disabledbackground='#12141F',
        )
        self.entry_box.pack(fill='x', ipady=10, padx=10, pady=4)
        self.entry_box.config(state='normal')

        self._entry_placeholder = 'Type a short sentence here…'
        self._entry_ph_active = True
        self.entry_box.insert(0, self._entry_placeholder)
        self.entry_box.config(fg='#4A4D60')
        self.entry_box.bind('<FocusIn>',  self._clear_entry_ph)
        self.entry_box.bind('<FocusOut>', self._restore_entry_ph)

        tk.Label(in_card, text='Review / Sentence  (multi-line)',
                 font=self.font_label,
                 bg=PALETTE['card'], fg=PALETTE['subtext']).pack(anchor='w', pady=(0, 6))

        txt_frame = tk.Frame(in_card, bg=PALETTE['border'],
                             highlightbackground=PALETTE['border'],
                             highlightthickness=1, bd=0)
        txt_frame.pack(fill='x')

        self.input_box = tk.Text(
            txt_frame,
            height=4,
            font=self.font_body,
            bg='#12141F',
            fg=PALETTE['text'],
            insertbackground=PALETTE['accent'],
            selectbackground=PALETTE['accent'],
            selectforeground='white',
            relief='flat',
            bd=0,
            wrap='word',
            padx=10, pady=10,
            state='normal',
            takefocus=True,
        )
        scrollbar = tk.Scrollbar(txt_frame, command=self.input_box.yview,
                                  bg=PALETTE['card'], troughcolor=PALETTE['bg'],
                                  relief='flat', bd=0)
        self.input_box.configure(yscrollcommand=scrollbar.set)
        self.input_box.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self._placeholder_text = 'Type or paste a movie review here…'
        self._show_placeholder()
        self.input_box.bind('<FocusIn>',  self._clear_placeholder)
        self.input_box.bind('<FocusOut>', self._restore_placeholder)

        btn_row = tk.Frame(root_frame, bg=PALETTE['bg'])
        btn_row.pack(fill='x', pady=16)

        self.analyse_btn = tk.Button(
            btn_row,
            text='  Analyse Sentiment  ',
            font=self.font_btn,
            bg=PALETTE['accent'],
            fg='white',
            activebackground=PALETTE['btn_hover'],
            activeforeground='white',
            relief='flat',
            bd=0,
            cursor='hand2',
            padx=24, pady=10,
            command=self._run_analysis,
        )
        self.analyse_btn.pack(side='left')

        self.clear_btn = tk.Button(
            btn_row,
            text='Clear',
            font=self.font_label,
            bg=PALETTE['card'],
            fg=PALETTE['subtext'],
            activebackground=PALETTE['border'],
            activeforeground=PALETTE['text'],
            relief='flat',
            bd=0,
            cursor='hand2',
            padx=14, pady=10,
            command=self._clear_all,
        )
        self.clear_btn.pack(side='left', padx=(10, 0))

        self._divider(root_frame)

        results_frame = tk.Frame(root_frame, bg=PALETTE['bg'])
        results_frame.pack(fill='both', expand=True, pady=(14, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)

        self.vader_card = self._result_card(results_frame, 'VADER', col=0)
        self.tb_card = self._result_card(results_frame, 'TextBlob', col=1)

        clean_frame = self._card(root_frame, fill='x', pady=(12, 0))

        tk.Label(clean_frame, text='Cleaned Input',
                 font=self.font_label,
                 bg=PALETTE['card'], fg=PALETTE['subtext']).pack(anchor='w', pady=(0, 4))

        self.clean_label = tk.Label(
            clean_frame,
            text='—',
            font=self.font_body,
            bg=PALETTE['card'],
            fg=PALETTE['subtext'],
            wraplength=680,
            justify='left',
            anchor='w',
        )
        self.clean_label.pack(fill='x')

        self.status_var = tk.StringVar(value='Ready')
        tk.Label(root_frame, textvariable=self.status_var,
                 font=self.font_sub,
                 bg=PALETTE['bg'], fg=PALETTE['subtext']).pack(anchor='e', pady=(8, 0))

    def _card(self, parent, **pack_kwargs):
        frame = tk.Frame(parent,
                         bg=PALETTE['card'],
                         highlightbackground=PALETTE['border'],
                         highlightthickness=1,
                         bd=0)
        frame.pack(**pack_kwargs)
        inner = tk.Frame(frame, bg=PALETTE['card'])
        inner.pack(fill='both', expand=True, padx=16, pady=14)
        return inner

    def _result_card(self, parent, title: str, col: int):
        outer = tk.Frame(parent,
                         bg=PALETTE['card'],
                         highlightbackground=PALETTE['border'],
                         highlightthickness=1,
                         bd=0)
        outer.grid(row=0, column=col, sticky='nsew',
                   padx=(0, 6) if col == 0 else (6, 0))

        inner = tk.Frame(outer, bg=PALETTE['card'])
        inner.pack(fill='both', expand=True, padx=18, pady=16)

        tk.Label(inner, text=title,
                 font=self.font_heading,
                 bg=PALETTE['card'], fg=PALETTE['text']).pack(anchor='w')

        accent_bar = tk.Frame(inner, bg=PALETTE['accent'], height=2)
        accent_bar.pack(fill='x', pady=(4, 12))

        badge_lbl = tk.Label(inner,
                              text='—',
                              font=self.font_badge,
                              bg=PALETTE['card'],
                              fg=PALETTE['subtext'])
        badge_lbl.pack(anchor='w')

        score_lbl = tk.Label(inner,
                               text='',
                               font=self.font_score,
                               bg=PALETTE['card'],
                               fg=PALETTE['subtext'])
        score_lbl.pack(anchor='w', pady=(2, 0))

        score_title = tk.Label(inner,
                                text='Score' if title == 'TextBlob' else 'Compound',
                                font=self.font_label,
                                bg=PALETTE['card'],
                                fg=PALETTE['subtext'])
        score_title.pack(anchor='w')

        return {'badge': badge_lbl, 'score': score_lbl}

    def _divider(self, parent):
        tk.Frame(parent, bg=PALETTE['border'], height=1).pack(fill='x')

    def _clear_entry_ph(self, _event=None):
        if getattr(self, '_entry_ph_active', False):
            self.entry_box.delete(0, 'end')
            self.entry_box.config(fg=PALETTE['text'])
            self._entry_ph_active = False

    def _restore_entry_ph(self, _event=None):
        if not self.entry_box.get().strip():
            self._entry_ph_active = True
            self.entry_box.insert(0, self._entry_placeholder)
            self.entry_box.config(fg='#4A4D60')

    def _show_placeholder(self):
        self.input_box.insert('1.0', self._placeholder_text)
        self.input_box.config(fg='#4A4D60')
        self._placeholder_active = True

    def _clear_placeholder(self, _event=None):
        if getattr(self, '_placeholder_active', False):
            self.input_box.delete('1.0', 'end')
            self.input_box.config(fg=PALETTE['text'])
            self._placeholder_active = False

    def _restore_placeholder(self, _event=None):
        if not self.input_box.get('1.0', 'end').strip():
            self._show_placeholder()

    def _run_analysis(self):
        entry_text = self.entry_box.get().strip()
        text_area  = self.input_box.get('1.0', 'end').strip()

        if entry_text and not getattr(self, '_entry_ph_active', False):
            raw = entry_text
        elif text_area and not getattr(self, '_placeholder_active', False):
            raw = text_area
        else:
            self._flash_status('⚠  Please enter some text first.')
            return

        self.analyse_btn.config(state='disabled', text='  Analysing…  ')
        self.status_var.set('Running analysis…')
        threading.Thread(target=self._analyse_worker, args=(raw,), daemon=True).start()

    def _analyse_worker(self, raw: str):
        cleaned = clean_text(raw)
        vader_label, vader_score   = get_vader_sentiment(cleaned)
        tb_label,    tb_score      = get_textblob_sentiment(cleaned)
        self.after(0, self._update_ui, cleaned, vader_label, vader_score, tb_label, tb_score)

    def _update_ui(self, cleaned, vader_label, vader_score, tb_label, tb_score):
        self.vader_card['badge'].config(text=vader_label,
                                        fg=LABEL_COLOR.get(vader_label, PALETTE['text']))
        self.vader_card['score'].config(text=f'{vader_score:+.4f}',
                                        fg=LABEL_COLOR.get(vader_label, PALETTE['text']))

        self.tb_card['badge'].config(text=tb_label,
                                     fg=LABEL_COLOR.get(tb_label, PALETTE['text']))
        self.tb_card['score'].config(text=f'{tb_score:+.4f}',
                                     fg=LABEL_COLOR.get(tb_label, PALETTE['text']))

        self.clean_label.config(text=cleaned or '(empty after cleaning)', fg=PALETTE['subtext'])

        self.analyse_btn.config(state='normal', text='  Analyse Sentiment  ')
        self.status_var.set(f'Done  ·  VADER: {vader_label}  ·  TextBlob: {tb_label}')

    def _clear_all(self):
        self.entry_box.delete(0, 'end')
        self._entry_ph_active = True
        self.entry_box.insert(0, self._entry_placeholder)
        self.entry_box.config(fg='#4A4D60')

        self.input_box.delete('1.0', 'end')
        self._show_placeholder()

        for card in (self.vader_card, self.tb_card):
            card['badge'].config(text='—', fg=PALETTE['subtext'])
            card['score'].config(text='', fg=PALETTE['subtext'])

        self.clean_label.config(text='—')
        self.status_var.set('Ready')

    def _flash_status(self, msg: str):
        self.status_var.set(msg)
        self.after(3000, lambda: self.status_var.set('Ready'))

if __name__ == '__main__':
    app = SentimentApp()
    app.mainloop()
