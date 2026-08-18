import { z } from 'zod';

export const uploadSchema = z.object({
  file: z
    .instanceof(File)
    .refine((file) => file.size <= 5 * 1024 * 1024, {
      message: 'File size must be less than 5MB',
    })
    .refine(
      (file) => ['image/jpeg', 'image/png', 'image/webp'].includes(file.type),
      {
        message: 'Please upload a JPEG, PNG, or WebP image',
      }
    ),
});

export type UploadFormData = z.infer<typeof uploadSchema>;
