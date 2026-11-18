# Runbook: fallas del pipeline de CI/CD

> Documento sintético con fines educativos. El pipeline descrito es un ejemplo, no el de una organización real.

## Etapas del pipeline

El pipeline de ejemplo tiene cinco etapas en este orden: `build`, `test`, `scan`, `publish` y `deploy`. Cada etapa
solo corre si la anterior terminó bien. La etapa `deploy` a producción requiere una aprobación manual registrada en el
pipeline.

## Clasificación inicial

Antes de reintentar, clasificar la falla según la etapa y el tipo de error:

- **Falla determinista**: el mismo commit falla siempre en el mismo punto (error de compilación, test roto, versión de
  dependencia incompatible). Reintentar no ayuda; hay que corregir el código o la configuración.
- **Falla intermitente (flaky)**: el mismo commit a veces pasa y a veces falla. Se permite un único reintento manual de
  la etapa; si vuelve a fallar, se trata como determinista y se abre un ticket para el test inestable.
- **Falla de infraestructura**: el agente de build se quedó sin disco, no pudo descargar dependencias o perdió la
  conexión con el registro de contenedores. Se resuelve en la plataforma, no en el código.

## Errores comunes por etapa

### build

- `No space left on device`: el agente acumuló imágenes de Docker. Limpiar el agente o usar agentes efímeros.
- Caché de dependencias corrupta: invalidar la clave de caché (por ejemplo, incluir el hash del lockfile en la clave) y
  volver a ejecutar. No borrar la caché de todos los pipelines a la vez.

### test

- Tests que dependen de la hora o del orden de ejecución: marcarlos como inestables y corregirlos en el mismo sprint.
- Tests que llaman a servicios externos reales: deben usar dobles de prueba o un servidor local.

### scan

- Una vulnerabilidad crítica en una dependencia bloquea el pipeline. No se desactiva el escaneo: se actualiza la
  dependencia o se registra una excepción temporal aprobada por seguridad, con fecha de vencimiento.

### publish

- `unauthorized: authentication required` al hacer push al registro de contenedores: la identidad del pipeline perdió
  permisos o el secreto de la conexión de servicio venció. Rotar el secreto en el almacén de secretos, nunca en el YAML
  del pipeline.

### deploy

- El despliegue a Kubernetes falla porque el rollout no completa: revisar las readiness probes de los pods nuevos
  (ver el runbook de AKS) y hacer rollback a la versión anterior si el servicio está degradado.
- Errores 429 de la API de Azure durante el despliegue: aplicar el runbook de throttling y no reintentar en bucle.

## Reglas de reintento

- Máximo un reintento manual por etapa y por commit.
- Nunca reintentar la etapa `deploy` a producción sin revisar primero por qué falló.
- Toda falla que bloquee `main` más de 30 minutos se comunica en el canal del equipo con el enlace a la ejecución.

## Evidencia a registrar

Para cada falla investigada se registra: identificador de la ejecución, commit, etapa, clasificación (determinista,
intermitente o infraestructura), causa encontrada y acción tomada.
