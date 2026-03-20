from IPython.display import display, HTML

def show_cms_event(event_id, preview_img=None):
    html_code = f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:12px; margin-top:10px;">

      <iframe src="https://cms3d.web.cern.ch/{event_id}/"
              width="100%" height="600"
              style="border:none;">
      </iframe>

    </div>
    """
    display(HTML(html_code))