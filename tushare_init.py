import tushare as ts

TOKEN = '1817835462588666beef4709678d38b18ae76ae4295a25e2d41c16e80af7db67'
API_URL = "http://8.136.22.187:8011/"

pro = ts.pro_api(TOKEN)
pro._DataApi__http_url = API_URL

def get_pro():
    return pro