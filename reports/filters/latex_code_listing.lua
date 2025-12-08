-- Turn paragraphs that start with "Code:" followed by a CodeBlock
-- into a captioned listings float (lstlisting with caption).

local function inline_to_plaintext(inlines)
  local s = pandoc.write(pandoc.Pandoc({ pandoc.Para(inlines) }), 'plain')
  return (s:gsub("\n+$", ""))
end

local function code_language(classes)
  if not classes or #classes == 0 then return nil end
  -- Use first class as language hint for listings
  return classes[1]
end

function Pandoc(doc)
  if FORMAT ~= 'latex' then return doc end
  local out = {}
  local i = 1
  while i <= #doc.blocks do
    local b = doc.blocks[i]
    if (b.t == 'Para' or b.t == 'Plain') and doc.blocks[i+1] and doc.blocks[i+1].t == 'CodeBlock' then
      local inl = b.c or b.content
      if inl and #inl >= 2 and inl[1].t == 'Str' and inl[1].text == 'Code:' then
        -- Build caption text from the rest of the paragraph
        local caption_inl = {}
        for j = 2, #inl do table.insert(caption_inl, inl[j]) end
        if #caption_inl > 0 and caption_inl[1].t == 'Space' then table.remove(caption_inl, 1) end
        local caption_tex = pandoc.write(pandoc.Pandoc({ pandoc.Plain(caption_inl) }), 'latex'):gsub("\n+", " ")
        local cb = doc.blocks[i+1]
        local code = cb.text or cb.c[1] or ''
        local lang = code_language((cb.attr or cb.attributes).classes)
        local opt = (lang and ("language=" .. lang .. ",") or "") .. "float=htbp,"
        local raw = table.concat({
          "\\begin{lstlisting}[" .. opt .. "caption={" .. caption_tex .. "}]",
          code,
          "\\end{lstlisting}"
        }, "\n")
        table.insert(out, pandoc.RawBlock('latex', raw))
        i = i + 2
      else
        table.insert(out, b)
        i = i + 1
      end
    else
      table.insert(out, b)
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end
