BEGIN;

CREATE TABLE IF NOT EXISTS global_embeddings (
    id UUID PRIMARY KEY,
    element_type VARCHAR(50) NOT NULL, 
    organization_id INTEGER NOT NULL,
    embedding VECTOR,
);

CREATE OR REPLACE FUNCTION sync_task_embedding_to_global()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NOT NULL THEN
        INSERT INTO global_embeddings (
            id,
            element_type,
            organization_id,
            embedding,
        )
        VALUES (
            NEW.id,
            'TASK',
            NEW.organization_id,
            NEW.embedding,
        )
        ON CONFLICT (id)
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            organization_id = EXCLUDED.organization_id,
            updated_at = now();
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE TRIGGER trigger_sync_task_embedding
AFTER INSERT OR UPDATE OF embedding ON tasks
FOR EACH ROW
EXECUTE FUNCTION sync_task_embedding_to_global();


COMMIT;

