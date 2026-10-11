interface Props {
  message: string;
  onRetry: () => void;
}

export function ErrorNotice({ message, onRetry }: Props) {
  return (
    <div className="error-notice" role="alert">
      <p>{message}</p>
      <button type="button" onClick={onRetry}>Tekrar dene</button>
    </div>
  );
}
