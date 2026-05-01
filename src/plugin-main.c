#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-spectrogram", "en-US")

extern struct obs_source_info spectrogram_source_info;

bool obs_module_load(void)
{
	obs_register_source(&spectrogram_source_info);
	return true;
}

void obs_module_unload(void) {}
