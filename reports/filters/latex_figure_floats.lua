-- Convert standalone images (with optional following "Figure:" paragraph)
-- into LaTeX floats (figure / figure*). Default width = \columnwidth.

local utils = require 'pandoc.utils'

local function parse_image_and_inline_caption(block)
  -- Returns (img, cap_inlines or nil) if block starts with an image.
  if not (block and (block.t == 'Para' or block.t == 'Plain')) then return nil end
  local inl = block.c or block.content
  if not inl or #inl == 0 then return nil end
  if inl[1].t ~= 'Image' then return nil end
  local img = inl[1]
  local cap = nil
  if #inl >= 3 and (inl[2].t == 'SoftBreak' or inl[2].t == 'LineBreak') and inl[3].t == 'Str' and inl[3].text == 'Figure:' then
    cap = {}
    for i = 4, #inl do table.insert(cap, inl[i]) end
    if #cap > 0 and cap[1].t == 'Space' then table.remove(cap, 1) end
  end
  return img, cap
end

local function image_src(img)
  -- Pandoc 3 API compatibility
  if img.src then return img.src end
  if img.target then return img.target[1] end
  return nil
end

local function image_classes(img)
  local attr = img.attr or img.attributes
  if attr and attr.classes then return attr.classes end
  return {}
end

local function caption_from_para(block)
  if not block or not (block.t == 'Para' or block.t == 'Plain') then return nil end
  local inl = block.c or block.content
  if not inl or #inl == 0 then return nil end
  -- Expect first token "Figure:" (case sensitive), then caption text
  local first = inl[1]
  if first.t == 'Str' and first.text == 'Figure:' then
    local cap = {}
    for i = 2, #inl do table.insert(cap, inl[i]) end
    -- Trim leading space if present
    if #cap > 0 and cap[1].t == 'Space' then table.remove(cap, 1) end
    return cap
  end
  return nil
end

local function latex_of_inlines(inlines)
  if not inlines then return nil end
  local doc = pandoc.Pandoc({ pandoc.Plain(inlines) })
  local s = pandoc.write(doc, 'latex') or ''
  -- collapse to single line inside \caption{...}
  s = s:gsub("\n+", " ")
  s = s:gsub("%s+$", "")
  return s
end

function Pandoc(doc)
  if FORMAT ~= 'latex' then return doc end
  local out = {}
  local i = 1
  while i <= #doc.blocks do
    local b = doc.blocks[i]
    local img, cap_inline = parse_image_and_inline_caption(b)
    if img then
      local src = image_src(img)
      local classes = image_classes(img)
      local wide = false
      for _,c in ipairs(classes) do
        if c == 'wide' or c == 'full' or c == 'figure*' or c == 'widefig' then wide = true end
      end
      local cap_next = caption_from_para(doc.blocks[i+1])
      local cap_inl = cap_inline or cap_next
      local cap_latex = latex_of_inlines(cap_inl)
      local env = wide and 'figure*' or 'figure'
      local width = wide and '\\textwidth' or '\\columnwidth'
      local gfx = string.format("\\pandocbounded{\\includegraphics[keepaspectratio,width=%s]{%s}}", width, src)
      local parts = { string.format("\\begin{%s}[htbp]", env), "\\centering", gfx }
      if cap_latex and #cap_latex > 0 then table.insert(parts, string.format("\\caption{%s}", cap_latex)) end
      table.insert(parts, string.format("\\end{%s}", env))
      table.insert(out, pandoc.RawBlock('latex', table.concat(parts, "\n")))
      -- Skip the caption paragraph if we consumed it
      if cap_inline then
        i = i + 1
      else
        i = i + (cap_next and 2 or 1)
      end
    else
      table.insert(out, b)
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end
