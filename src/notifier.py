"""
Windows Native Toast Notification Helper for Ripleytia AI Studio.
Sends beautiful desktop notifications when long tasks (Cover, Batch, Dataset, Training) complete.
"""

import os
import sys
import subprocess
import threading


def send_windows_notification(title: str, message: str, sound: bool = True):
    """
    Sends a native Windows Toast notification using PowerShell.
    Runs asynchronously in a background daemon thread to never block UI.
    """
    def _worker():
        try:
            # Escape strings for PowerShell
            safe_title = title.replace('"', '`"').replace("'", "''")
            safe_msg = message.replace('"', '`"').replace("'", "''")
            
            # PowerShell script using Windows Runtime Toast Notifications
            ps_script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

$template = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>{safe_title}</text>
            <text>{safe_msg}</text>
        </binding>
    </visual>
    <audio src="ms-winsoundevent:Notification.Default" silent="{'false' if sound else 'true'}" />
</toast>
"@

$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Ripleytia AI Studio").Show($toast)
"""
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            # Fallback to simple console notice if toast fails
            print(f"[Notifier Uyarısı] Bildirim gönderilemedi: {e}")

    threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    send_windows_notification("Ripleytia AI Studio", "Test bildirimi başarıyla çalışıyor! 🎵")
