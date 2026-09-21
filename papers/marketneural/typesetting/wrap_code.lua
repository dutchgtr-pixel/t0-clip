-- Permit line breaks in long source identifiers without changing copied text.
function Code(el)
  local escaped = el.text:gsub('\\', '\\textbackslash{}')
    :gsub('([#$%%&_{}])', '\\%1'):gsub('~','\\textasciitilde{}'):gsub('%^','\\textasciicircum{}')
  escaped = escaped:gsub('\\_', '\\_\\allowbreak{}'):gsub('/', '/\\allowbreak{}')
  return pandoc.RawInline('latex', '\\texttt{' .. escaped .. '}')
end
