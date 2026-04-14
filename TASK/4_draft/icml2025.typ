// Файл: icml2025.typ

#let icml2025(
  title: "",
  short-title: none,
  authors: (),
  abstract: none,
  accepted: false,
  body,
) = {
  // Настройка базовых параметров страницы
  let running-title = if short-title != none { short-title } else { title }
  
  set page(
    paper: "us-letter",
    margin: (left: 0.75in, top: 1.0in, right: 1.0in, bottom: 1.0in),
    header-ascent: 10pt,
    header: context {
      let page-num = counter(page).get().first()
      if page-num > 1 {
        set text(size: 9pt, weight: "bold")
        align(center)[#running-title]
        v(-0.05in)
        line(length: 100%, stroke: 1pt)
      }
    },
    footer: context {
      if not accepted and counter(page).get().first() == 1 {
        set text(size: 9pt)
        line(length: 0.8in, stroke: 0.5pt)
        v(2pt)
        [Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.]
      } else if accepted and counter(page).get().first() == 1 {
        set text(size: 9pt)
        line(length: 0.8in, stroke: 0.5pt)
        v(2pt)
        [#emph[Proceedings of the $42^(n d)$ International Conference on Machine Learning], Vancouver, Canada, PMLR 267, 2025. Copyright 2025 by the author(s).]
      }
    }
  )

  // Настройка шрифта и абзацев
  set text(font: "Times New Roman", size: 10pt)
  set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
  
  // Межстрочный интервал (spacing между абзацами)
  show par: set block(spacing: 11pt)

  // Форматирование заголовка статьи
  align(center)[
    #line(length: 100%, stroke: 1pt)
    #v(0.1in)
    #text(size: 14pt, weight: "bold")[#title]
    #v(0.1in)
    #line(length: 100%, stroke: 1pt)
    #v(0.3in) // Отступ до авторов или начала текста
  ]

  // Блок авторов (отображается только если accepted: true)
  if accepted and authors.len() > 0 {
    align(center)[
      #text(size: 10pt, weight: "bold")[
        #authors.map(a => a.name + super(a.affil)).join(", ", last: " and ")
      ]
      #v(0.1in)
    ]
  }

  // Переход в двухколоночный режим
  show: columns.with(2, gutter: 0.25in)

  // Настройка заголовков (1, 2 и 3 уровней)
  set heading(numbering: "1.1.1")
  
  show heading.where(level: 1): it => block(
    above: 0.25in, 
    below: 0.15in, 
    text(size: 11pt, weight: "bold")[
      #counter(heading).display() #it.body
    ]
  )

  show heading.where(level: 2): it => block(
    above: 0.2in, 
    below: 0.13in, 
    text(size: 10pt, weight: "bold")[
      #counter(heading).display() #it.body
    ]
  )

  show heading.where(level: 3): it => block(
    above: 0.18in, 
    below: 0.1in, 
    text(size: 10pt)[ // Small caps эмулируется через uppercase/шрифт в Typst
      #smallcaps[#counter(heading).display() #it.body]
    ]
  )

  // Аннотация (Abstract)
  if abstract != none {
    align(center)[
      #text(size: 11pt, weight: "bold")[Abstract]
    ]
    pad(x: 0.25in)[
      #text(size: 10pt)[#abstract]
    ]
    v(0.4in)
  }

  // Настройка подписей к рисункам и таблицам
  show figure.caption: it => text(size: 9pt)[#it]

  // Основной текст документа
  body
}