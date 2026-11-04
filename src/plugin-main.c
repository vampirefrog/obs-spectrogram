#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-spectrogram", "en-US")

extern struct obs_source_info spectrogram_source_info;

MODULE_EXPORT const char *obs_module_name(void)
{
	return "Spectrogram";
}

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Rolling spectrogram source (FFT or Resonate analysis)";
}

bool obs_module_load(void)
{
	obs_register_source(&spectrogram_source_info);
	blog(LOG_INFO, "[obs-spectrogram] version %s loaded from %s",
	     PLUGIN_VERSION, obs_get_module_binary_path(obs_current_module()));
	return true;
}

void obs_module_unload(void) {}
