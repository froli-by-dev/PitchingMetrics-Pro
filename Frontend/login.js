const API_URL = "http://127.0.0.1:8000"; // O el puerto donde corra tu FastAPI

document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault(); // Evita que la página se recargue

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const errorMsg = document.getElementById('error-msg');

    // FastAPI espera los datos en formato FormData (x-www-form-urlencoded)
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

try {
  const response = await fetch(`${API_URL}/analizar-video/${pitcherId}`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) throw new Error("Error en el servidor al procesar el video");

  const data = await response.json();
  console.log("Respuesta recibida del Backend:", data); // Para inspeccionar la estructura real

  // 1. Cargar y Reproducir Video Anotado
  const videoPath = data.video_procesado_url || data.video_url || "";
  if (videoPath) {
    const videoFilename = videoPath.split('/').pop();
    const videoSrc = `${API_URL}/videos_procesados/${videoFilename}?t=${new Date().getTime()}`;

    player.src = videoSrc;
    player.muted = true;
    player.load();
    player.play().catch(err => console.log("Error de reproducción:", err));
  }

  // 2. Extraer fases biomecánicas según la estructura devuelta
  const fases = data.fases_biomecanicas || data.fases || data;
  const footStrikeData = fases.foot_strike || fases.footStrike || {};
  const releasePointData = fases.release_point || fases.releasePoint || {};

  // 3. Renderizar Métricas
  renderMetrics(footStrikeData, fsContainer);
  renderMetrics(releasePointData, relContainer);

} catch (err) {
  alert("Ocurrió un error al procesar los datos en la interfaz.");
  console.error("Detalle del error:", err);
}
});