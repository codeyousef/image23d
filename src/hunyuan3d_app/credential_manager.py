"""Secure credential management with UI components"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import gradio as gr

logger = logging.getLogger(__name__)


@dataclass
class CredentialInfo:
    """Information about a stored credential"""
    service: str
    username: Optional[str]
    description: str
    required: bool = False
    url: Optional[str] = None


class CredentialManager:
    """Manages secure storage and retrieval of API credentials"""
    
    # Known services and their info
    KNOWN_SERVICES = {
        "huggingface": CredentialInfo(
            service="huggingface",
            username="token",
            description="Hugging Face API token for accessing gated models",
            required=False,
            url="https://huggingface.co/settings/tokens"
        ),
        "civitai": CredentialInfo(
            service="civitai",
            username="api_key",
            description="Civitai API key for model downloads",
            required=False,
            url="https://civitai.com/user/account"
        ),
        "openai": CredentialInfo(
            service="openai",
            username="api_key",
            description="OpenAI API key for GPT-based features",
            required=False,
            url="https://platform.openai.com/api-keys"
        ),
        "stability": CredentialInfo(
            service="stability",
            username="api_key",
            description="Stability AI API key",
            required=False,
            url="https://platform.stability.ai/account/keys"
        ),
        "replicate": CredentialInfo(
            service="replicate",
            username="api_token",
            description="Replicate API token for cloud inference",
            required=False,
            url="https://replicate.com/account/api-tokens"
        )
    }
    
    def __init__(self, app_name: str = "Hunyuan3DStudio"):
        self.app_name = app_name
        self.keyring_available = self._check_keyring_available()
        
        # Fallback to encrypted file storage if keyring not available
        self.fallback_file = Path.home() / ".hunyuan3d" / "credentials.enc"
        self.fallback_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _check_keyring_available(self) -> bool:
        """Check if keyring is available and working"""
        try:
            import keyring
            # Test keyring by trying to get a dummy credential
            keyring.get_password(self.app_name, "_test")
            return True
        except Exception as e:
            logger.warning(f"Keyring not available: {e}")
            return False
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for fallback storage"""
        key_file = self.fallback_file.parent / ".key"
        
        if key_file.exists():
            return key_file.read_bytes()
        else:
            # Generate new key
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            # Set restrictive permissions
            if os.name != 'nt':  # Unix-like systems
                os.chmod(key_file, 0o600)
            return key
    
    def store_credential(
        self,
        service: str,
        credential: str,
        username: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Store a credential securely
        
        Args:
            service: Service name (e.g., 'huggingface')
            credential: The credential to store
            username: Optional username/identifier
            
        Returns:
            Tuple of (success, message)
        """
        if not credential:
            return False, "Empty credential provided"
            
        try:
            # Get service info
            service_info = self.KNOWN_SERVICES.get(service)
            if service_info:
                username = username or service_info.username
            
            if self.keyring_available:
                import keyring
                keyring.set_password(self.app_name, f"{service}:{username or 'default'}", credential)
                logger.info(f"Stored credential for {service} in system keyring")
            else:
                # Use encrypted file storage
                self._store_credential_fallback(service, username, credential)
                logger.info(f"Stored credential for {service} in encrypted file")
                
            # Also set environment variable for current session
            env_vars = {
                "huggingface": "HF_TOKEN",
                "openai": "OPENAI_API_KEY",
                "civitai": "CIVITAI_API_KEY",
                "stability": "STABILITY_API_KEY",
                "replicate": "REPLICATE_API_TOKEN"
            }
            
            if service in env_vars:
                os.environ[env_vars[service]] = credential
                
            return True, f"✅ Credential for {service} stored successfully"
            
        except Exception as e:
            logger.error(f"Error storing credential: {e}")
            return False, f"❌ Error storing credential: {str(e)}"
    
    def _store_credential_fallback(self, service: str, username: Optional[str], credential: str):
        """Store credential in encrypted file (fallback method)"""
        from cryptography.fernet import Fernet
        
        # Load existing credentials
        credentials = {}
        if self.fallback_file.exists():
            try:
                key = self._get_encryption_key()
                f = Fernet(key)
                encrypted_data = self.fallback_file.read_bytes()
                decrypted_data = f.decrypt(encrypted_data)
                credentials = json.loads(decrypted_data.decode())
            except Exception as e:
                logger.error(f"Error loading existing credentials: {e}")
                
        # Update credentials
        key_name = f"{service}:{username or 'default'}"
        credentials[key_name] = credential
        
        # Encrypt and save
        key = self._get_encryption_key()
        f = Fernet(key)
        encrypted_data = f.encrypt(json.dumps(credentials).encode())
        self.fallback_file.write_bytes(encrypted_data)
        
        # Set restrictive permissions
        if os.name != 'nt':  # Unix-like systems
            os.chmod(self.fallback_file, 0o600)
    
    def get_credential(
        self,
        service: str,
        username: Optional[str] = None
    ) -> Optional[str]:
        """Retrieve a stored credential
        
        Args:
            service: Service name
            username: Optional username/identifier
            
        Returns:
            The credential or None if not found
        """
        try:
            # Check environment variable first
            env_vars = {
                "huggingface": "HF_TOKEN",
                "openai": "OPENAI_API_KEY",
                "civitai": "CIVITAI_API_KEY",
                "stability": "STABILITY_API_KEY",
                "replicate": "REPLICATE_API_TOKEN"
            }
            
            if service in env_vars and env_vars[service] in os.environ:
                return os.environ[env_vars[service]]
            
            # Get service info
            service_info = self.KNOWN_SERVICES.get(service)
            if service_info:
                username = username or service_info.username
                
            if self.keyring_available:
                import keyring
                credential = keyring.get_password(self.app_name, f"{service}:{username or 'default'}")
                return credential
            else:
                # Use encrypted file storage
                return self._get_credential_fallback(service, username)
                
        except Exception as e:
            logger.error(f"Error retrieving credential: {e}")
            return None
    
    def _get_credential_fallback(self, service: str, username: Optional[str]) -> Optional[str]:
        """Get credential from encrypted file (fallback method)"""
        if not self.fallback_file.exists():
            return None
            
        try:
            from cryptography.fernet import Fernet
            
            key = self._get_encryption_key()
            f = Fernet(key)
            encrypted_data = self.fallback_file.read_bytes()
            decrypted_data = f.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            key_name = f"{service}:{username or 'default'}"
            return credentials.get(key_name)
            
        except Exception as e:
            logger.error(f"Error retrieving credential from fallback: {e}")
            return None
    
    def delete_credential(
        self,
        service: str,
        username: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Delete a stored credential
        
        Args:
            service: Service name
            username: Optional username/identifier
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get service info
            service_info = self.KNOWN_SERVICES.get(service)
            if service_info:
                username = username or service_info.username
                
            if self.keyring_available:
                import keyring
                keyring.delete_password(self.app_name, f"{service}:{username or 'default'}")
            else:
                # Use encrypted file storage
                self._delete_credential_fallback(service, username)
                
            # Remove from environment if present
            env_vars = {
                "huggingface": "HF_TOKEN",
                "openai": "OPENAI_API_KEY",
                "civitai": "CIVITAI_API_KEY",
                "stability": "STABILITY_API_KEY",
                "replicate": "REPLICATE_API_TOKEN"
            }
            
            if service in env_vars and env_vars[service] in os.environ:
                del os.environ[env_vars[service]]
                
            return True, f"✅ Credential for {service} deleted successfully"
            
        except Exception as e:
            logger.error(f"Error deleting credential: {e}")
            return False, f"❌ Error deleting credential: {str(e)}"
    
    def _delete_credential_fallback(self, service: str, username: Optional[str]):
        """Delete credential from encrypted file (fallback method)"""
        if not self.fallback_file.exists():
            return
            
        try:
            from cryptography.fernet import Fernet
            
            # Load existing credentials
            key = self._get_encryption_key()
            f = Fernet(key)
            encrypted_data = self.fallback_file.read_bytes()
            decrypted_data = f.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            # Remove credential
            key_name = f"{service}:{username or 'default'}"
            if key_name in credentials:
                del credentials[key_name]
                
            # Save updated credentials
            if credentials:
                encrypted_data = f.encrypt(json.dumps(credentials).encode())
                self.fallback_file.write_bytes(encrypted_data)
            else:
                # Delete file if no credentials left
                self.fallback_file.unlink()
                
        except Exception as e:
            logger.error(f"Error deleting credential from fallback: {e}")
    
    def list_stored_services(self) -> List[str]:
        """List all services with stored credentials
        
        Returns:
            List of service names
        """
        services = set()
        
        # Check environment variables
        env_map = {
            "HF_TOKEN": "huggingface",
            "OPENAI_API_KEY": "openai",
            "CIVITAI_API_KEY": "civitai",
            "STABILITY_API_KEY": "stability",
            "REPLICATE_API_TOKEN": "replicate"
        }
        
        for env_var, service in env_map.items():
            if env_var in os.environ:
                services.add(service)
                
        # Check keyring or fallback
        if self.keyring_available:
            try:
                import keyring
                # This is platform-specific and may not work on all systems
                # Fallback to known services check
                for service in self.KNOWN_SERVICES:
                    if self.get_credential(service):
                        services.add(service)
            except:
                pass
        else:
            # Check fallback file
            if self.fallback_file.exists():
                try:
                    from cryptography.fernet import Fernet
                    
                    key = self._get_encryption_key()
                    f = Fernet(key)
                    encrypted_data = self.fallback_file.read_bytes()
                    decrypted_data = f.decrypt(encrypted_data)
                    credentials = json.loads(decrypted_data.decode())
                    
                    for key_name in credentials:
                        service = key_name.split(":")[0]
                        services.add(service)
                except:
                    pass
                    
        return list(services)
    
    def create_ui_component(self) -> gr.Group:
        """Create Gradio UI component for credential management
        
        Returns:
            Gradio Group component
        """
        with gr.Group() as credential_group:
            gr.Markdown("### 🔐 API Credentials Management")
            gr.Markdown(
                "Securely store your API keys. Credentials are encrypted and stored in your system's keyring when available."
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Service selection
                    service_choices = list(self.KNOWN_SERVICES.keys())
                    service_dropdown = gr.Dropdown(
                        choices=service_choices,
                        value=service_choices[0] if service_choices else None,
                        label="Service",
                        interactive=True
                    )
                    
                    # Credential input
                    credential_input = gr.Textbox(
                        label="API Key / Token",
                        type="password",
                        placeholder="Enter your API key or token",
                        interactive=True
                    )
                    
                    # Action buttons
                    with gr.Row():
                        save_btn = gr.Button("💾 Save Credential", variant="primary")
                        delete_btn = gr.Button("🗑️ Delete Credential", variant="secondary")
                        test_btn = gr.Button("🧪 Test Credential")
                        
                with gr.Column(scale=2):
                    # Service info
                    service_info = gr.Markdown("")
                    
                    # Stored services
                    stored_services = gr.Markdown("")
                    
            # Status message
            status_msg = gr.Markdown("")
            
            # Update service info when selection changes
            def update_service_info(service):
                if service in self.KNOWN_SERVICES:
                    info = self.KNOWN_SERVICES[service]
                    info_text = f"""
### {service.title()} Credentials

**Description:** {info.description}

**Required:** {'Yes' if info.required else 'No'}

**Get your API key:** [Click here]({info.url})
"""
                    return info_text
                return ""
            
            # Update stored services list
            def update_stored_services():
                services = self.list_stored_services()
                if services:
                    service_list = "\n".join([f"- ✅ {s}" for s in services])
                    return f"### Stored Credentials\n\n{service_list}"
                return "### Stored Credentials\n\nNo credentials stored yet."
            
            # Save credential handler
            def save_credential(service, credential):
                if not service:
                    return "❌ Please select a service"
                if not credential:
                    return "❌ Please enter a credential"
                    
                success, message = self.store_credential(service, credential)
                
                # Clear the input for security
                return message
            
            # Delete credential handler
            def delete_credential(service):
                if not service:
                    return "❌ Please select a service"
                    
                success, message = self.delete_credential(service)
                return message
            
            # Test credential handler
            def test_credential(service):
                if not service:
                    return "❌ Please select a service"
                    
                credential = self.get_credential(service)
                if credential:
                    # Truncate for display
                    truncated = credential[:8] + "..." + credential[-4:] if len(credential) > 16 else "***"
                    return f"✅ Credential found for {service}: {truncated}"
                else:
                    return f"❌ No credential found for {service}"
            
            # Wire up events
            service_dropdown.change(
                update_service_info,
                inputs=[service_dropdown],
                outputs=[service_info]
            )
            
            save_btn.click(
                save_credential,
                inputs=[service_dropdown, credential_input],
                outputs=[status_msg]
            ).then(
                lambda: "",
                outputs=[credential_input]  # Clear input
            ).then(
                update_stored_services,
                outputs=[stored_services]
            )
            
            delete_btn.click(
                delete_credential,
                inputs=[service_dropdown],
                outputs=[status_msg]
            ).then(
                update_stored_services,
                outputs=[stored_services]
            )
            
            test_btn.click(
                test_credential,
                inputs=[service_dropdown],
                outputs=[status_msg]
            )
            
            # Initial updates
            credential_group.load(
                update_stored_services,
                outputs=[stored_services]
            )
            
            if service_choices:
                credential_group.load(
                    update_service_info,
                    inputs=[service_dropdown],
                    outputs=[service_info]
                )
                
        return credential_group