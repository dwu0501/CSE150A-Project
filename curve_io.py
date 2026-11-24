"""Small, diff-friendly JSON exports: one checkpoint per line, no rounding."""
import json


def save_curve(data, path):
    settings = {k: v for k, v in data.items() if k != 'rows'}
    fields = [json.dumps(k) + ': ' + json.dumps(v, allow_nan=False) for k, v in settings.items()]
    rows = ',\n'.join('    ' + json.dumps(row, allow_nan=False) for row in data['rows'])
    fields.append('"rows": [\n' + rows + '\n  ]')
    path.write_text('{\n  ' + ',\n  '.join(fields) + '\n}\n', encoding='utf-8')
