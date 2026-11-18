# Runbook: throttling de la API de Azure (HTTP 429)

> Documento sintético con fines educativos. Los límites numéricos son una política interna de ejemplo, no los límites
> oficiales de Azure.

## Qué es el throttling

Azure Resource Manager y muchos servicios de Azure limitan la cantidad de solicitudes por intervalo de tiempo. Cuando
un cliente supera el límite, la API responde `HTTP 429 Too Many Requests`. No es un error de permisos ni de
configuración: es una señal de que el cliente debe esperar antes de volver a intentar.

## Síntomas

- Logs de automatizaciones (Terraform, scripts de inventario, pipelines de despliegue) con `429 TooManyRequests` o
  mensajes `The request is being throttled`.
- Respuestas que incluyen el header `Retry-After` con la cantidad de segundos a esperar.
- Headers `x-ms-ratelimit-remaining-subscription-reads` o `x-ms-ratelimit-remaining-subscription-writes` con valores
  cercanos a cero.

## Acciones inmediatas

1. **Respetar `Retry-After`.** Si la respuesta trae ese header, esperar exactamente esos segundos antes del siguiente
   intento.
2. **Backoff exponencial con jitter** cuando no hay `Retry-After`: esperar 2, 4, 8 y 16 segundos, con una variación
   aleatoria de hasta 1 segundo, y como máximo 5 reintentos por solicitud.
3. **Reducir la concurrencia.** Bajar el paralelismo de la automatización que genera la carga; por ejemplo, ejecutar
   Terraform con `-parallelism=5` en lugar del valor por defecto.
4. **Identificar al consumidor.** Revisar en el Activity Log qué identidad (service principal o managed identity)
   concentra las solicitudes.

## Qué no hacer

- No reintentar en un bucle sin espera: cada reintento inmediato consume cuota y prolonga el throttling.
- No crear identidades nuevas para "repartir" la cuota de una misma suscripción.
- No ocultar el error: un 429 persistente es un problema de diseño de la automatización.

## Prevención

- Cachear lecturas repetidas (por ejemplo, el inventario de recursos) en lugar de consultarlas en cada ejecución.
- Usar operaciones por lotes cuando el servicio las ofrece.
- Programar los trabajos de inventario masivo fuera de la ventana de despliegues.
- Política interna de ejemplo: ninguna automatización debe superar 20 solicitudes por segundo contra una misma
  suscripción.

## Escalamiento

Si el throttling dura más de 15 minutos con la concurrencia ya reducida, abrir un incidente de severidad 3 y avisar al
equipo dueño de la automatización identificada.
