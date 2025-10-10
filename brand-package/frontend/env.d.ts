/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SUPABASE_URL: "https://pmzxlnpmapqlwsphiath.supabase.co"
  readonly VITE_SUPABASE_ANON_KEY: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBtenhsbnBtYXBxbHdzcGhpYXRoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkzOTYwMDUsImV4cCI6MjA3NDk3MjAwNX0.lmLmBgEz0O5dfKeu_4HwOxNMJq0IrXVFiSAKKCs2OCM"
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}