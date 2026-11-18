"""Builds the report on top of the faculty template, following its formatting guidelines:
A4, Times New Roman, chapter 16 pt bold uppercase centred, section 14 pt bold, subsection 12 pt bold,
body 12 pt justified 1.5 spacing 6 pt after, captions 11 pt bold, tables 11 pt, code Consolas 10 single,
margins L 1.5" R/T/B 1", figures/tables/equations numbered per chapter, roman then arabic page numbers."""
import copy
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
TEXT_WIDTH_IN = 8.27 - 1.5 - 1.0


def _font(style_or_run, name=FONT, size=None, bold=None, italic=None):
    f = style_or_run.font
    f.name = name
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    rpr = style_or_run.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), name)


def _outline(style, level):
    ppr = style.element.get_or_add_pPr()
    el = OxmlElement("w:outlineLvl")
    el.set(qn("w:val"), str(level))
    ppr.append(el)


def _field(paragraph, instr, placeholder="Right-click and choose Update Field."):
    """Insert a complex field (TOC, PAGE, ...) that Word fills in when fields are updated."""
    def run_with(child):
        r = OxmlElement("w:r")
        r.append(child)
        paragraph._p.append(r)
        return r
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); run_with(b)
    t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = f" {instr} "; run_with(t)
    s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate"); run_with(s)
    ph = OxmlElement("w:t"); ph.text = placeholder; run_with(ph)
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); run_with(e)


class Report:
    def __init__(self, template_path, members, title, faculty):
        self.doc = Document(template_path)
        self.chapter_no = 0
        self.fig_no = 0
        self.tab_no = 0
        self.eq_no = 0
        self._first_prelim = True
        self._styles()
        self._cover(members, title)
        self._tidy_front(faculty)
        self._strip_after_bonafide()

    # ---------- setup ----------
    def _styles(self):
        st = self.doc.styles

        def para_style(name, size, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6,
                       spacing=1.5, outline=None, keep_next=False, font=FONT, italic=False):
            s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            s.base_style = st["Normal"]
            _font(s, font, size, bold, italic)
            pf = s.paragraph_format
            pf.alignment = align
            pf.space_before = Pt(before)
            pf.space_after = Pt(after)
            if spacing == 1:
                pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            else:
                pf.line_spacing = spacing
            pf.keep_with_next = keep_next
            pf.first_line_indent = Inches(0)
            pf.left_indent = Inches(0)
            if outline is not None:
                _outline(s, outline)
            return s

        para_style("RptBody", 12)
        para_style("RptChapterNo", 16, True, WD_ALIGN_PARAGRAPH.CENTER, 0, 6, keep_next=True)
        para_style("RptChapter", 16, True, WD_ALIGN_PARAGRAPH.CENTER, 0, 18, outline=0, keep_next=True)
        para_style("RptPrelim", 16, True, WD_ALIGN_PARAGRAPH.CENTER, 0, 18, outline=0, keep_next=True)
        para_style("RptSection", 14, True, WD_ALIGN_PARAGRAPH.LEFT, 12, 6, outline=1, keep_next=True)
        para_style("RptSub", 12, True, WD_ALIGN_PARAGRAPH.LEFT, 6, 6, outline=2, keep_next=True)
        para_style("RptFigCaption", 11, True, WD_ALIGN_PARAGRAPH.CENTER, 4, 10, spacing=1)
        para_style("RptTabCaption", 11, True, WD_ALIGN_PARAGRAPH.CENTER, 8, 4, spacing=1, keep_next=True)
        para_style("RptFigure", 12, False, WD_ALIGN_PARAGRAPH.CENTER, 6, 0, spacing=1, keep_next=True)
        para_style("RptTable", 11, False, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, spacing=1)
        para_style("RptEquation", 12, False, WD_ALIGN_PARAGRAPH.LEFT, 4, 4, spacing=1.15, italic=False)
        code = para_style("RptCode", 9, False, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, spacing=1, font="Consolas")
        code.paragraph_format.left_indent = Inches(0.08)
        shd = OxmlElement("w:shd")  # light grey band behind code listings
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
        code.element.get_or_add_pPr().append(shd)
        para_style("RptBullet", 12)
        para_style("RptRef", 12, align=WD_ALIGN_PARAGRAPH.LEFT)
        for cap in ("RptTabCaption", "RptFigCaption", "RptEquation"):  # never break a caption/equation mid-way
            st[cap].paragraph_format.keep_together = True
        eq = st["RptEquation"].paragraph_format
        eq.tab_stops.add_tab_stop(Inches(TEXT_WIDTH_IN / 2), WD_TAB_ALIGNMENT.CENTER)
        eq.tab_stops.add_tab_stop(Inches(TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT)
        bl = st["RptBullet"].paragraph_format
        bl.left_indent = Inches(0.3)
        bl.first_line_indent = Inches(-0.2)
        rf = st["RptRef"].paragraph_format
        rf.left_indent = Inches(0.4)
        rf.first_line_indent = Inches(-0.4)
        for lvl in (1, 2, 3):  # table-of-contents entries in Times New Roman
            _font(st[f"toc {lvl}"], FONT, 12)
        names = [s.name for s in st]
        toc9 = st["toc 9"] if "toc 9" in names else st.add_style("toc 9", WD_STYLE_TYPE.PARAGRAPH)
        _font(toc9, FONT, 12, bold=False)  # lists of figures/tables use level 9: plain, no indent
        toc9.paragraph_format.left_indent = Inches(0)
        toc9.paragraph_format.space_after = Pt(4)

    def _cover(self, members, title):
        from docx.text.paragraph import Paragraph
        # free room for 5 names on the cover and the longer bonafide text: drop some blank spacer lines
        kids = list(self.doc.element.body)
        for i in (9, 10, 15, 22, 25) + tuple(range(47, 55)):
            el = kids[i]
            blank = el.tag == qn("w:p") and not "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
            if blank and next(el.iter(qn("w:drawing")), None) is None and next(el.iter(qn("w:sectPr")), None) is None:
                el.getparent().remove(el)
        for p in self.doc.paragraphs:
            if p.text.startswith("Design, Implementation, and Performance Evaluation"):
                self._set_text(p, title)
                break
        name_ps = [p for p in self.doc.paragraphs if re.fullmatch(r"\s*Name\s+\[Reg No\]\s*", p.text)]
        first, second = name_ps[0], name_ps[1]
        self._set_text(first, "{} [{}]".format(*members[0]))
        for n, r in members[1:]:  # one line per member, cloned from the template's name line
            new = copy.deepcopy(second._p)
            second._p.addprevious(new)
            self._set_text(Paragraph(new, self.doc._body), f"{n} [{r}]")
        second._p.getparent().remove(second._p)
        for p in self.doc.paragraphs:
            if "STUDENT1 NAME" in p.text:
                names = ", ".join(f"{n.upper()} [{r}]" for n, r in members)
                self._set_text(p, "Certified that 21CSC305P - MACHINE LEARNING project report is the bonafide work of "
                                  f"“{names}” who carried out the task of completing the project within the "
                                  "allotted time.")
                break

    @staticmethod
    def _set_text(p, text):
        runs = p.runs
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)

    def _tidy_front(self, faculty):
        """Cover and bonafide: replace the faculty name, drop the template's ad-hoc indents and padding spaces,
        centre every text line, and lay out the certificate as one justified block."""
        body = self.doc.element.body
        tbl = next(el for el in body if el.tag == qn("w:tbl"))
        fixes = {"requirementsfor": "requirements for", "ANDTECHNOLOGY": "AND TECHNOLOGY",
                 "NOV  2026": "NOV 2026", "KATTANKULATHUR- 603 203": "KATTANKULATHUR – 603 203"}
        for t in tbl.iter(qn("w:t")):  # signature table
            if t.text and "Prince" in t.text:
                t.text = faculty
        for p in self.doc.paragraphs:
            if p._p.getparent() is not body or list(body).index(p._p) > list(body).index(tbl):
                continue
            text = p.text
            if not text.strip():
                continue
            new = " ".join(text.split())
            for a, b in fixes.items():
                new = new.replace(a, b)
            if "Prince" in new:
                new = faculty
            self._set_text(p, new)  # always merge runs: some template spaces sit in condensed runs and vanish
            pf = p.paragraph_format
            pf.left_indent = pf.right_indent = pf.first_line_indent = Inches(0)
            if new.startswith("Certified that"):
                pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                pf.line_spacing = 1.5
                pf.space_before = Pt(12)
            else:
                pf.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def _strip_after_bonafide(self):
        body = self.doc.element.body
        kids = list(body)
        tables = [i for i, el in enumerate(kids) if el.tag == qn("w:tbl")]
        start = tables[0] + 1  # the signature table ends the bonafide page
        for el in kids[start:]:
            if el.tag != qn("w:sectPr"):
                body.remove(el)
        # close the bonafide page with its own section so it keeps the template margins
        p = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        ppr.append(copy.deepcopy(body.find(qn("w:sectPr"))))
        p.append(ppr)
        body.find(qn("w:sectPr")).addprevious(p)

    # ---------- text ----------
    def _runs(self, p, text, size=None):
        """Minimal inline markup: **bold**, *italic*, ^{sup}, _{sub}."""
        for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]*\}|_\{[^}]*\})", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = p.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("*"):
                r = p.add_run(tok[1:-1]); r.italic = True
            elif tok.startswith("^{"):
                r = p.add_run(tok[2:-1]); r.font.superscript = True
            elif tok.startswith("_{"):
                r = p.add_run(tok[2:-1]); r.font.subscript = True
            else:
                r = p.add_run(tok)
            if size:
                r.font.size = Pt(size)
        return p

    def p(self, text):
        para = self._runs(self.doc.add_paragraph(style="RptBody"), text)
        if "https://" in text:  # long unbreakable URLs make justified lines gappy
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        return para

    def bullets(self, items):
        for it in items:
            self._runs(self.doc.add_paragraph(style="RptBullet"), "• " + it)

    def page_break(self):
        self.doc.add_paragraph(style="RptBody").add_run().add_break(WD_BREAK.PAGE)

    def prelim(self, title):
        p = self.doc.add_paragraph(title, style="RptPrelim")
        p.paragraph_format.page_break_before = not self._first_prelim  # first one opens a new section anyway
        self._first_prelim = False

    def chapter(self, title):
        self.chapter_no += 1
        self.fig_no = self.tab_no = self.eq_no = 0
        if self.chapter_no == 1:  # arabic page numbers start here
            self.doc.add_section(WD_SECTION.NEW_PAGE)
        p = self.doc.add_paragraph(f"CHAPTER {self.chapter_no}", style="RptChapterNo")
        p.paragraph_format.page_break_before = self.chapter_no > 1
        self.doc.add_paragraph(f"{self.chapter_no}. {title.upper()}", style="RptChapter")

    def unnumbered_chapter(self, title):
        self.doc.add_paragraph(title.upper(), style="RptChapter").paragraph_format.page_break_before = True

    def h2(self, text):
        self.doc.add_paragraph(text, style="RptSection")

    def h3(self, text):
        self.doc.add_paragraph(text, style="RptSub")

    def eq(self, text):
        self.eq_no += 1
        p = self.doc.add_paragraph(style="RptEquation")
        p.add_run("\t")
        self._runs(p, text)
        p.add_run(f"\t({self.chapter_no}.{self.eq_no})")
        return f"({self.chapter_no}.{self.eq_no})"

    # ---------- figures and tables ----------
    def fig(self, path, caption, width=TEXT_WIDTH_IN):
        self.fig_no += 1
        label = f"Figure {self.chapter_no}.{self.fig_no}"
        p = self.doc.add_paragraph(style="RptFigure")
        p.add_run().add_picture(str(path), width=Inches(min(width, TEXT_WIDTH_IN)))
        self.doc.add_paragraph(f"{label}: {caption}", style="RptFigCaption")
        return label

    def table(self, header, rows, caption, widths=None, align_numbers=True):
        label = None
        if caption:  # caption=None gives an unnumbered table (abbreviations)
            self.tab_no += 1
            label = f"Table {self.chapter_no}.{self.tab_no}"
            self.doc.add_paragraph(f"{label}: {caption}", style="RptTabCaption")
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        widths = widths or [TEXT_WIDTH_IN / len(header)] * len(header)
        for i, row in enumerate([header] + [list(r) for r in rows]):
            cells = t.rows[i].cells
            for j, val in enumerate(row):
                c = cells[j]
                c.width = Inches(widths[j])
                c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                para = c.paragraphs[0]
                para.style = self.doc.styles["RptTable"]
                para.paragraph_format.space_before = para.paragraph_format.space_after = Pt(2)
                txt = str(val)
                self._runs(para, txt, size=11)
                # one rule for every table: header centred, first (label) column left, value columns centred
                centre = i == 0 or (align_numbers and j > 0)
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER if centre else WD_ALIGN_PARAGRAPH.LEFT
                if i == 0:
                    for r in para.runs:
                        r.bold = True
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "DCE9F4")
                    c._tc.get_or_add_tcPr().append(shd)
            if i == 0:  # repeat header row on each page
                trpr = t.rows[0]._tr.get_or_add_trPr()
                h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trpr.append(h)
        for k, row in enumerate(t.rows):  # rows never split; short tables stay on one page with their caption
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            if len(t.rows) <= 16 and k < len(t.rows) - 1:
                for c in row.cells:
                    c.paragraphs[0].paragraph_format.keep_with_next = True
        self.doc.add_paragraph(style="RptBody").paragraph_format.space_after = Pt(2)
        return label

    # ---------- fields and code ----------
    def toc(self):
        # only our own heading styles: the template cover uses Heading 1-4 for its title block
        _field(self.doc.add_paragraph(style="RptBody"),
               'TOC \\h \\z \\t "RptPrelim,1,RptChapter,1,RptSection,2,RptSub,3"')

    def list_of(self, caption_style):
        _field(self.doc.add_paragraph(style="RptBody"), f'TOC \\h \\z \\t "{caption_style},9"')  # toc 9: plain style

    def code(self, text):
        for line in text.splitlines() or [""]:
            p = self.doc.add_paragraph(style="RptCode")
            p.add_run(line.replace("\t", "    ") if line else " ")

    # ---------- page numbers and save ----------
    def _footer_page_number(self, section, fmt, start):
        section.footer.is_linked_to_previous = False
        p = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
        p.text = ""
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _field(p, "PAGE", "1")
        for r in p.runs:
            _font(r, FONT, 12)
        pg = section._sectPr.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType")
            section._sectPr.append(pg)
        pg.set(qn("w:fmt"), fmt)
        if start is not None:
            pg.set(qn("w:start"), str(start))

    def finish(self, path):
        secs = self.doc.sections
        for s in secs[2:]:  # cover (0) and bonafide (1) keep the template layout
            s.left_margin, s.right_margin = Inches(1.5), Inches(1.0)
            s.top_margin, s.bottom_margin = Inches(1.0), Inches(1.0)
        self._footer_page_number(secs[1], "lowerRoman", 2)
        self._footer_page_number(secs[2], "lowerRoman", None)
        self._footer_page_number(secs[-1], "decimal", 1)
        self.doc.save(path)
