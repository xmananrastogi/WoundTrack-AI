import os

html_path = 'templates/index.html'
css_path = 'static/css/style.css'
js_path = 'static/js/main.js'

with open(html_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Open files
with open(css_path, 'w', encoding='utf-8') as css_f, open(js_path, 'w', encoding='utf-8') as js_f:
    new_html = []
    
    in_style = False
    in_script = False
    script_started_at = 0
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Style extraction
        if line_num == 9: # <style>
            in_style = True
            new_html.append('  <link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/style.css\') }}">\n')
            continue
        elif line_num == 417: # </style>
            in_style = False
            continue
            
        # Script extraction (Line 1178 is the <script> where app JS starts)
        elif line_num == 1178:
            in_script = True
            new_html.append('<script src="{{ url_for(\'static\', filename=\'js/main.js\') }}"></script>\n')
            continue
        elif line_num == 2164:
            in_script = False
            continue
            
        # Routing lines
        if in_style:
            css_f.write(line)
        elif in_script:
            js_f.write(line)
        else:
            new_html.append(line)

with open(html_path, 'w', encoding='utf-8') as f:
    f.writelines(new_html)

print("Extraction complete.")
