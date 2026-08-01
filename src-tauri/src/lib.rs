pub mod app;
pub mod codec;

use tauri::Manager;
use tauri_plugin_single_instance::init as init_single_instance;

use app::state::AppState;

pub fn run() {
    tauri::Builder::default()
        .plugin(init_single_instance(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            let state =
                AppState::new(data_dir.join("openxlate.db")).map_err(std::io::Error::other)?;
            let gateway_state = state.gateway_state();
            app.manage(state);
            tauri::async_runtime::spawn(async move {
                app::gateway::serve(gateway_state).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app::commands::translate_text,
            app::commands::get_settings,
            app::commands::update_settings,
            app::commands::detect_language,
            app::commands::get_supported_languages,
            app::commands::list_providers,
            app::commands::create_provider,
            app::commands::update_provider,
            app::commands::delete_provider,
            app::commands::get_gateway_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running OpenXlate");
}
