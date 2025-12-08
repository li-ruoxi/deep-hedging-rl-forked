-- Fix accidentally escaped LaTeX equation environments that appear as
-- literal text (e.g., "\\begin{equation}") by converting them into
-- raw LaTeX blocks with single backslashes.

local function inline_text(inlines)
  local t = {}
  for _,el in ipairs(inlines) do
    if el.t == 'Str' then table.insert(t, el.text)
    elseif el.t == 'Space' then table.insert(t, ' ')
    elseif el.t == 'SoftBreak' or el.t == 'LineBreak' then table.insert(t, '\n')
    else
      -- ignore formatting in equations; this filter is for raw TeX runs only
    end
  end
  return table.concat(t)
end

local function fix_run(blocks, i)
  -- returns (newBlock, newIndex) if an equation run is found starting at i
  local b = blocks[i]
  if not (b.t == 'Para' or b.t == 'Plain') then return nil end
  local s = inline_text(b.c or b.content)
  if not s:match('^%s*\\\\begin%{equation%}') then return nil end
  local pieces = { s }
  local j = i + 1
  while j <= #blocks do
    local bb = blocks[j]
    if not (bb.t == 'Para' or bb.t == 'Plain') then break end
    local sj = inline_text(bb.c or bb.content)
    table.insert(pieces, sj)
    if sj:match('\\\\end%{equation%}') then break end
    j = j + 1
  end
  local raw = table.concat(pieces, '\n')
  -- collapse double backslashes to single backslashes
  raw = raw:gsub('\\\\', '\\')
  return pandoc.RawBlock('latex', raw), j + 1
end

function Pandoc(doc)
  if FORMAT ~= 'latex' then return doc end
  local out = {}
  local i = 1
  while i <= #doc.blocks do
    local newb, nxt = fix_run(doc.blocks, i)
    if newb then
      table.insert(out, newb)
      i = nxt
    else
      table.insert(out, doc.blocks[i])
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end

