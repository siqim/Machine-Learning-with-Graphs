# Created by 知乎@小鱼干
# Link: https://www.zhihu.com/question/25430912/answer/957067212

import sys
import re

interline_tag = '\n<img src="https://www.zhihu.com/equation?tex={}" alt="{}\\\\" class="ee_img tr_noresize" eeimg="1">\n'
interline_pattern = "\$\$\n*(.*?)\n*\$\$"
inline_tag = '<img src="https://www.zhihu.com/equation?tex={}" alt="{}" class="ee_img tr_noresize" eeimg="1">'
inline_pattern = "\$\n*(.*?)\n*\$"

def replace_tex(content):
	def dashrepl(matchobj, tag):
		formular = matchobj.group(1)
		return tag.format(formular, formular)

	content = re.sub(interline_pattern, lambda mo: dashrepl(mo, interline_tag), content)
	content = re.sub(inline_pattern, lambda mo: dashrepl(mo, inline_tag), content)

	return content

if __name__=='__main__':
	assert len(sys.argv) > 1, "Error: need filename as a argument"
	filename = sys.argv[1]
	with open(filename, 'r', encoding='utf-8') as f:
		content = f.read()
	with open(filename, 'w', encoding='utf-8') as f:
		f.write(replace_tex(content))
