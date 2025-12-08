-- Convert Pandoc tables to inline tabular environments.
local function convert_table(tbl)
  local latex = pandoc.write(pandoc.Pandoc({tbl}), 'latex')

  latex = latex:gsub('\\begin%{longtable%}%[%]', '\\begin{longtable}')
  latex = latex:gsub('\\endfirsthead%s*', '')
  latex = latex:gsub('\\endhead%s*', '')
  latex = latex:gsub('\\endfoot%s*', '')
  latex = latex:gsub('\\endlastfoot%s*', '')
  latex = latex:gsub('\\caption%b{}', '')
  latex = latex:gsub('\\label%b{}', '')

  latex = latex:gsub('\\begin%{longtable%}(%b{})', function(spec)
    return '\\begin{tabular}' .. spec
  end)
  latex = latex:gsub('\\end%{longtable%}', '\\end{tabular}')

  return latex
end

function Table(tbl)
  if FORMAT ~= 'latex' then
    return nil
  end

  local body = convert_table(tbl)
  local wrapped = table.concat({
    '{\\small\\begin{center}',
    body,
    '\\end{center}}'
  }, '\n')

  return pandoc.RawBlock('latex', wrapped)
end
